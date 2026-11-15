"""Lanjutkan pelatihan model CIFAR-10 dengan Stochastic Weight Averaging (SWA).

Berangkat dari bobot model yang sudah ada (ResNet-18, `.h5`), skrip ini meneruskan
SGD dengan learning rate tetap dan merata-ratakan bobot di akhir setiap epoch
(Izmailov dkk., 2018). Statistik BatchNorm model rata-rata dihitung ulang dari
himpunan latih.

Pembagian data sama dengan train.py: 45.000 gambar latih dan 5.000 validasi
(seed 42). Model yang dipakai dan kebijakan tampilan (TTA) dipilih dari akurasi
validasi. Himpunan uji baru dibaca setelah pilihan itu dibekukan.

Model dasar v5 tidak lagi ada di pohon kerja (aplikasi memakai hasil skrip ini).
Ambil dari riwayat git, lalu jalankan dari akar repositori:

    git show 72895f7:model_cnn_cifar10_v5.h5 > model_cnn_cifar10_v5.h5
    python scripts/finetune_swa.py --time-budget-minutes 270
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch

import train
from classify import TTA_POLICIES, tta_views


def cpu_has_bf16() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return False
    return "avx512_bf16" in flags or "amx_bf16" in flags


def load_keras_weights(path: Path):
    """Bobot `.h5` ke ResNet-18 PyTorch. Kebalikan dari train.export_keras.

    Berkas dibaca dengan h5py, bukan TensorFlow: memuat TensorFlow sebelum
    pelatihan PyTorch dalam satu proses bisa bentrok. TensorFlow baru dimuat
    saat ekspor di akhir, seperti di train.py.
    """
    import h5py

    model = train.CifarResNet18()
    with h5py.File(path, "r") as handle:
        weights = handle["model_weights"]

        def read(layer: str, name: str) -> torch.Tensor:
            return torch.from_numpy(np.array(weights[f"{layer}/{layer}/{name}"], dtype=np.float32))

        def conv(module, layer):
            module.weight.data.copy_(read(layer, "kernel").permute(3, 2, 0, 1))

        def bn(module, layer):
            module.weight.data.copy_(read(layer, "gamma"))
            module.bias.data.copy_(read(layer, "beta"))
            module.running_mean.copy_(read(layer, "moving_mean"))
            module.running_var.copy_(read(layer, "moving_variance"))

        conv(model.stem_conv, "stem_conv")
        bn(model.stem_bn, "stem_bn")
        stages = (model.layer1, model.layer2, model.layer3, model.layer4)
        for stage_index, stage in enumerate(stages, start=1):
            for block_index, block in enumerate(stage):
                prefix = f"s{stage_index}b{block_index}"
                conv(block.conv1, f"{prefix}_conv1")
                bn(block.bn1, f"{prefix}_bn1")
                conv(block.conv2, f"{prefix}_conv2")
                bn(block.bn2, f"{prefix}_bn2")
                if block.shortcut is not None:
                    conv(block.shortcut[0], f"{prefix}_proj")
                    bn(block.shortcut[1], f"{prefix}_proj_bn")
        model.fc.weight.data.copy_(read("predictions", "kernel").T)
        model.fc.bias.data.copy_(read("predictions", "bias"))

        # Rata-rata dan varians kanal disimpan di konfigurasi lapisan, bukan di bobot.
        config = json.loads(handle.attrs["model_config"])
    layers = {layer["config"]["name"]: layer["config"] for layer in config["config"]["layers"]}
    mean = np.array(layers["standardize"]["mean"], dtype=np.float32).reshape(-1)
    std = np.sqrt(np.array(layers["standardize"]["variance"], dtype=np.float32).reshape(-1))
    if mean.shape != (3,) or std.shape != (3,):
        raise RuntimeError("Lapisan standardize tidak berisi tiga kanal.")
    model.eval()
    return model.to(memory_format=torch.channels_last), mean, std


@torch.inference_mode()
def predict_policy(model, images, mean, std, policy: str, batch_size: int) -> np.ndarray:
    """Kelas tebakan untuk gambar uint8 NCHW, memakai tampilan dari classify.py."""
    model.eval()
    views = len(TTA_POLICIES[policy])
    step = max(1, batch_size // views)
    predicted = np.empty(images.shape[0], dtype=np.int64)
    for start in range(0, images.shape[0], step):
        chunk = images[start : start + step]
        nhwc = chunk.permute(0, 2, 3, 1).float().div(255.0).numpy()
        batch = torch.from_numpy(tta_views(nhwc, policy)).permute(0, 3, 1, 2)
        batch = train.normalize(batch, mean, std).contiguous(memory_format=torch.channels_last)
        probs = torch.softmax(model(batch).float(), dim=1).double()
        averaged = probs.view(views, chunk.shape[0], -1).mean(dim=0)
        predicted[start : start + chunk.shape[0]] = averaged.argmax(dim=1).numpy()
    return predicted


def correct_count(model, images, labels, mean, std, policy: str, batch_size: int) -> int:
    predicted = predict_policy(model, images, mean, std, policy, batch_size)
    return int((predicted == labels.numpy()).sum())


@torch.no_grad()
def update_bn(model, images, mean, std, batch_size: int) -> None:
    """Hitung ulang statistik BatchNorm dari satu lintasan latih beraugmentasi."""
    layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    momenta = [layer.momentum for layer in layers]
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None  # rata-rata kumulatif atas semua batch
    model.train()
    order = torch.randperm(images.shape[0])
    for start in range(0, images.shape[0], batch_size):
        index = order[start : start + batch_size]
        batch = train.augment_batch(images[index].float().div(255.0))
        batch = train.apply_cutout(train.normalize(batch, mean, std))
        model(batch.contiguous(memory_format=torch.channels_last))
    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum
    model.eval()


class WeightAverage:
    """Rata-rata berjalan atas parameter model (bukan buffer BatchNorm)."""

    def __init__(self):
        self.count = 0
        self.params: dict[str, torch.Tensor] = {}

    def add(self, model) -> None:
        self.count += 1
        for name, param in model.named_parameters():
            value = param.detach().double()
            if name not in self.params:
                self.params[name] = value.clone()
            else:
                self.params[name] += (value - self.params[name]) / self.count

    def build(self, template, images, mean, std, batch_size: int):
        model = copy.deepcopy(template)
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(self.params[name].to(param.dtype))
        update_bn(model, images, mean, std, batch_size)
        return model


def choose(scores: dict[str, dict[str, int]]) -> tuple[str, str]:
    """Skor validasi tertinggi; seri dimenangkan tampilan paling sedikit, lalu model lama."""
    ranked = sorted(
        ((count, -len(TTA_POLICIES[policy]), name == "base", name, policy)
         for name, by_policy in scores.items()
         for policy, count in by_policy.items()),
        reverse=True,
    )
    _, _, _, name, policy = ranked[0]
    return name, policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(ROOT / "model_cnn_cifar10_v5.h5"))
    parser.add_argument("--output", default=str(ROOT / "model_cnn_cifar10_v6.h5"))
    parser.add_argument("--metrics", default=str(ROOT / "results" / "swa_metrics.json"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--swa-start", type=int, default=5,
                        help="Epoch pertama (1-indeks) yang ikut dirata-ratakan.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-per-class", type=int, default=train.VAL_PER_CLASS)
    parser.add_argument("--allow-split-mismatch", action="store_true",
                        help="Hanya untuk uji asap dengan data tiruan.")
    args = parser.parse_args()
    started = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)
    images, labels = train.load_train_arrays(data_dir)
    train_index, val_index = train.stratified_split(labels, args.val_per_class, args.seed)
    split_mean, split_std = train.channel_mean_std(images[train_index])

    base, mean, std = load_keras_weights(Path(args.base))
    # Rata-rata kanal ikut tersimpan di model. Jika sama, pembagian datanya sama,
    # sehingga validasi di sini memang tidak pernah dilihat model dasar.
    same_split = np.allclose(split_mean, mean, atol=1e-6) and np.allclose(split_std, std, atol=1e-6)
    print(f"Pembagian data sama dengan model dasar: {same_split}")
    if not same_split and not args.allow_split_mismatch:
        raise RuntimeError("Pembagian latih/validasi berbeda dari model dasar.")

    train_x = torch.from_numpy(np.ascontiguousarray(images[train_index]))
    train_y = torch.from_numpy(labels[train_index])
    val_x = torch.from_numpy(np.ascontiguousarray(images[val_index]))
    val_y = torch.from_numpy(labels[val_index])
    del images, labels
    mean_t = torch.from_numpy(mean).view(1, 3, 1, 1)
    std_t = torch.from_numpy(std).view(1, 3, 1, 1)
    eval_batch = args.batch_size * 4
    bf16 = cpu_has_bf16()
    print(
        f"Train {len(train_index)} gambar, validasi {len(val_index)} gambar, "
        f"{torch.get_num_threads()} thread, bfloat16={bf16}"
    )

    def val_scores(model) -> dict[str, int]:
        return {
            policy: correct_count(model, val_x, val_y, mean_t, std_t, policy, eval_batch)
            for policy in TTA_POLICIES
        }

    base_scores = val_scores(base)
    print(f"Validasi model dasar (benar dari {len(val_index)}): {base_scores}")

    model = copy.deepcopy(base)
    optimizer = train.make_optimizer(model)
    for group in optimizer.param_groups:
        group["lr"] = args.lr
    average = WeightAverage()
    history = []
    epochs_done = 0
    longest = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        loss, acc = train.run_epoch(
            model, optimizer, train_x, train_y, mean_t, std_t, args.batch_size, bf16=bf16
        )
        sgd_val = correct_count(model, val_x, val_y, mean_t, std_t, "flip", eval_batch)
        if epoch >= args.swa_start:
            average.add(model)
        entry = {
            "epoch": epoch,
            "train_loss": round(loss, 6),
            "train_acc": round(acc, 6),
            "sgd_val_flip": sgd_val,
            "averaged": average.count,
        }
        line = (
            f"epoch={epoch}/{args.epochs} lr={args.lr} train_loss={loss:.4f} "
            f"train_acc={acc:.4f} sgd_val_flip={sgd_val} rata2={average.count}"
        )
        if average.count and epoch % args.eval_every == 0:
            swa = average.build(model, train_x, mean_t, std_t, args.batch_size)
            entry["swa_val_flip"] = correct_count(
                swa, val_x, val_y, mean_t, std_t, "flip", eval_batch
            )
            line += f" swa_val_flip={entry['swa_val_flip']}"
        elapsed = time.time() - epoch_started
        longest = max(longest, elapsed)
        entry["seconds"] = round(elapsed, 1)
        history.append(entry)
        epochs_done = epoch
        print(f"{line} sec={elapsed:.1f}", flush=True)
        if args.time_budget_minutes is not None:
            # Berhenti sebelum epoch yang tidak akan sempat selesai.
            if time.time() - started + 1.2 * longest > args.time_budget_minutes * 60:
                print("Batas waktu tercapai; rata-rata bobot dibekukan di sini.")
                break

    if average.count == 0:
        raise RuntimeError("Belum ada epoch yang dirata-ratakan.")
    swa = average.build(model, train_x, mean_t, std_t, args.batch_size)
    scores = {"base": base_scores, "swa": val_scores(swa)}
    print(f"Validasi model SWA (benar dari {len(val_index)}): {scores['swa']}")
    chosen_name, chosen_policy = choose(scores)
    chosen = swa if chosen_name == "swa" else base
    print(f"Dipilih dari validasi: model={chosen_name} tampilan={chosen_policy}")

    output = Path(args.output)
    # Pilihan sudah beku. Baru sekarang himpunan uji dibaca.
    test_x_np, test_y_np = train.load_test_arrays(data_dir)
    test_x = torch.from_numpy(np.ascontiguousarray(test_x_np))
    test_y = torch.from_numpy(test_y_np)
    test = {
        policy: correct_count(chosen, test_x, test_y, mean_t, std_t, policy, eval_batch)
        for policy in dict.fromkeys(("single", "flip", chosen_policy))
    }
    total = int(test_y.shape[0])
    for policy, count in test.items():
        print(f"Uji {policy}: {count}/{total} = {count / total:.4f}")

    payload = {
        "method": "SWA fine-tune",
        "base_model": Path(args.base).name,
        "same_split_as_base": bool(same_split),
        "seed": args.seed,
        "train_images": len(train_index),
        "validation_images": len(val_index),
        "test_images": total,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs_completed": epochs_done,
        "epochs_averaged": average.count,
        "swa_start_epoch": args.swa_start,
        "bfloat16": bf16,
        "framework": f"PyTorch {torch.__version__}",
        "seconds": round(time.time() - started, 1),
        "validation_correct": scores,
        "selected_by": "validation_correct",
        "selected_model": chosen_name,
        "selected_policy": chosen_policy,
        "selected_model_file": output.name if chosen_name == "swa" else Path(args.base).name,
        "test_set_used_for_training": False,
        "test_set_used_for_selection": False,
        "test_correct_selected_model": test,
        "history": history,
    }
    metrics = Path(args.metrics)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Metrik ditulis ke {metrics}")

    # TensorFlow baru dimuat di sini, setelah semua perhitungan PyTorch selesai.
    # Ekspor memeriksa bahwa keluaran Keras sama dengan PyTorch.
    if chosen_name == "swa":
        train.export_keras(chosen, mean, std, output)


if __name__ == "__main__":
    main()
