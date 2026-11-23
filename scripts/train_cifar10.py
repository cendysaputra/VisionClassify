"""Fine-tune ViT-Small/16 on CIFAR-10 and export an INT8 ONNX model.

Not part of the Streamlit runtime. Needs a GPU and training packages:

    pip install torch torchvision transformers onnx onnxruntime
    python scripts/train_cifar10.py --output model_vit_cifar10.onnx

To keep test evaluation honest (AGENTS.md rule 8), this script only loads
``data_batch_1`` through ``data_batch_5`` (50,000 training images) and takes
a stratified validation split (default 5,000 images) for checkpoint selection.
``test_batch`` is never read here; measure test accuracy once afterwards with:

    python scripts/eval_cifar_test.py --write-results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classify import CLASS_NAMES, MODEL_FILENAME, MODEL_INPUT_SIZE  # noqa: E402

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
# Pin the pretrained backbone to an exact revision so runs are reproducible.
BASE_MODEL = "WinKawaks/vit-small-patch16-224"
BASE_MODEL_REVISION = "0c08c1829024fa177bd69b6dd09b05ad980854e1"
BASE_MODEL_LICENSE = "apache-2.0"
TRAIN_BATCH_FILES: tuple[str, ...] = tuple(f"data_batch_{index}" for index in range(1, 6))


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cifar10_train_batches(root: Path) -> Path:
    """Download and extract CIFAR-10, verifying the archive checksum."""
    folder = root / "cifar-10-batches-py"
    if all((folder / name).is_file() for name in TRAIN_BATCH_FILES):
        return folder
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    if not archive.is_file():
        print(f"Downloading CIFAR-10 from {CIFAR10_URL}")
        urllib.request.urlretrieve(CIFAR10_URL, archive)
    actual_md5 = _hash_file(archive, "md5")
    if actual_md5 != CIFAR10_MD5:
        raise RuntimeError(f"CIFAR-10 archive MD5 mismatch: {actual_md5} != {CIFAR10_MD5}")
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    return folder


def load_cifar10_train(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only the 50,000 official training images (never test_batch)."""
    folder = ensure_cifar10_train_batches(root)
    image_parts, label_parts = [], []
    for name in TRAIN_BATCH_FILES:
        with (folder / name).open("rb") as handle:
            entry = pickle.load(handle, encoding="bytes")
        image_parts.append(entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        label_parts.append(np.asarray(entry[b"labels"], dtype=np.int64))
    images = np.concatenate(image_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    if images.shape != (50000, 32, 32, 3) or labels.shape != (50000,):
        raise RuntimeError(f"Unexpected train shape: {images.shape} {labels.shape}")
    return images, labels


def stratified_split_indices(
    labels: np.ndarray,
    val_size: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split training indices into class-balanced train and validation subsets."""
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("labels must be a non-empty 1D array")
    classes = np.unique(y)
    if val_size <= 0 or val_size >= y.size or val_size % len(classes) != 0:
        raise ValueError(
            f"val_size ({val_size}) must be positive, smaller than {y.size}, "
            f"and divisible by {len(classes)} classes"
        )
    per_class = val_size // len(classes)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in classes:
        positions = np.flatnonzero(y == cls)
        if positions.size <= per_class:
            raise ValueError(f"Class {cls} has too few samples ({positions.size}) for val split")
        shuffled = rng.permutation(positions)
        val_idx.append(shuffled[:per_class])
        train_idx.append(shuffled[per_class:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def build_provenance_record(
    *,
    output_path: Path,
    model_sha256: str,
    model_bytes: int,
    train_size: int,
    val_size: int,
    best_val_accuracy: float,
    best_epoch: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> dict[str, object]:
    """Build a reproducible training provenance record matching AGENTS.md rules 8-9."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/train_cifar10.py",
        "model_file": output_path.name,
        "model_sha256": model_sha256,
        "model_bytes": model_bytes,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_model_license": BASE_MODEL_LICENSE,
        "pretraining_data": "ImageNet-1k (WinKawaks/vit-small-patch16-224)",
        "dataset": "CIFAR-10",
        "train_images": train_size,
        "val_images": val_size,
        "test_set_used_for_training": False,
        "test_set_used_for_selection": False,
        "best_val_accuracy": round(best_val_accuracy, 6),
        "best_epoch": best_epoch,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "input_size": MODEL_INPUT_SIZE,
            "quantization": "dynamic_int8_quint8",
        },
        "device": device,
    }


def _preprocess_array(raw: np.ndarray, *, flip: bool = False) -> np.ndarray:
    """Match classify.preprocess exactly so ONNX parity holds at inference."""
    image = Image.fromarray(raw)
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    resized = image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)
    arr = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(arr.transpose(2, 0, 1), dtype=np.float32)


def train_and_export(args: argparse.Namespace) -> None:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from torch.utils.data import DataLoader, Dataset
    from transformers import ViTForImageClassification, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cpu":
        print(
            "WARNING: running on CPU; fine-tuning ViT-Small/16 at 224x224 is slow without a GPU.",
            flush=True,
        )

    images, labels = load_cifar10_train(Path(args.data_dir))
    train_idx, val_idx = stratified_split_indices(labels, val_size=args.val_size, seed=args.seed)

    class _CifarSplit(Dataset):
        def __init__(self, indices: np.ndarray, augment: bool):
            self.indices = indices
            self.augment = augment

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, pos: int):
            idx = int(self.indices[pos])
            flip = self.augment and random.random() < 0.5
            tensor = torch.from_numpy(_preprocess_array(images[idx], flip=flip))
            return tensor, int(labels[idx])

    train_loader = DataLoader(
        _CifarSplit(train_idx, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        _CifarSplit(val_idx, augment=False),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    id2label = {index: name.lower() for index, name in enumerate(CLASS_NAMES)}
    label2id = {name: index for index, name in id2label.items()}
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL,
        revision=BASE_MODEL_REVISION,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(train_loader, start=1):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(pixel_values=batch_x).logits
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.item())
            if step % 100 == 0 or step == len(train_loader):
                print(
                    f"[epoch {epoch}/{args.epochs}] step {step}/{len(train_loader)} "
                    f"loss={running_loss / step:.4f}",
                    flush=True,
                )

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                preds = model(pixel_values=batch_x).logits.argmax(dim=1).cpu().numpy()
                correct += int((preds == batch_y.numpy()).sum())
        val_acc = correct / len(val_idx)
        print(
            f"[epoch {epoch}/{args.epochs}] val_accuracy={val_acc:.4f} ({correct}/{len(val_idx)})",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    model.cpu().eval()

    class _LogitsOnly(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.inner(pixel_values=pixel_values).logits

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = output_path.with_suffix(".fp32.onnx")
    dummy = torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, dtype=torch.float32)
    try:
        torch.onnx.export(
            _LogitsOnly(model),
            dummy,
            str(fp32_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
        )
        # QUInt8 dynamic quantization matches the CPUExecutionProvider kernels
        # used in classify.py and keeps the model well below the 100 MB git limit.
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
        )
    finally:
        fp32_path.unlink(missing_ok=True)

    digest = _hash_file(output_path, "sha256")
    record = build_provenance_record(
        output_path=output_path,
        model_sha256=digest,
        model_bytes=output_path.stat().st_size,
        train_size=len(train_idx),
        val_size=len(val_idx),
        best_val_accuracy=best_val_acc,
        best_epoch=best_epoch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=str(device),
    )
    provenance_path = Path(args.provenance_out)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path} ({output_path.stat().st_size / 1e6:.2f} MB, sha256={digest})")
    print(f"wrote {provenance_path} (best_val_accuracy={best_val_acc:.4f} at epoch {best_epoch})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / MODEL_FILENAME))
    parser.add_argument(
        "--provenance-out",
        default=str(ROOT / "results" / "provenance" / "train_run.json"),
    )
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train_and_export(args)


if __name__ == "__main__":
    main()
