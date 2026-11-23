"""Train a CIFAR-10 classifier on pinned DINOv2-Reg features and export ONNX.

Designed to run reproducibly both locally and sharded across GitHub Actions
CPU runners (no GPU required). Strictly follows AGENTS.md rules 8-14:

- Reads ONLY CIFAR-10 ``data_batch_1``..``data_batch_5`` (50,000 training
  images). ``test_batch`` is never loaded, read, or referenced here.
- Splits the 50,000 training images deterministically into 45,000 train and
  5,000 stratified validation images (``seed=42``). All model/regularization
  choices are made strictly on the 5,000-image validation split.
- Exports a standalone ONNX model (< 100 MB) that accepts the exact ``[-1, 1]``
  224x224 input produced by ``classify.preprocess`` and ``verify-model.yml``.
- Writes full provenance to ``results/provenance/train_run.json`` with
  ``test_set_used_for_training: false`` and ``test_set_used_for_selection: false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classify import MODEL_FILENAME, MODEL_INPUT_SIZE  # noqa: E402
from train_cifar10 import load_cifar10_train, stratified_split_indices  # noqa: E402

BASE_MODEL = "facebook/dinov2-with-registers-base"
BASE_MODEL_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
BASE_MODEL_LICENSE = "apache-2.0"
PRETRAINING_DATA = "LVD-142M (self-supervised, no CIFAR-10 labels)"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 768 * 5  # Last 4 CLS tokens + final spatial patch mean
MAX_MODEL_BYTES = 99_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_slice(total: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    """Return [start, end) indices partitioning ``total`` across ``num_shards``."""
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise ValueError(f"Invalid shard {shard_index}/{num_shards}")
    base, rem = divmod(total, num_shards)
    start = shard_index * base + min(shard_index, rem)
    end = start + base + (1 if shard_index < rem else 0)
    return start, end


def preprocess_batch(raw_images: np.ndarray, *, flip: bool = False) -> np.ndarray:
    """Match classify.preprocess and eval_cifar_test.preprocess_batch exactly."""
    batch = np.empty((len(raw_images), 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)
    for index, raw in enumerate(raw_images):
        image = Image.fromarray(raw)
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        resized = image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)
        arr = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
        batch[index] = arr.transpose(2, 0, 1)
    return batch


def fold_standard_scaler(
    weight: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold ``(x - mean) / scale`` directly into a linear layer's weight and bias."""
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    m = np.asarray(mean, dtype=np.float64)
    s = np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)
    w_folded = w / s[np.newaxis, :]
    b_folded = b - w_folded @ m
    return w_folded.astype(np.float32), b_folded.astype(np.float32)


def _bake_dinov2_pos_embed(model: object, img_size: int = MODEL_INPUT_SIZE):
    """Pre-interpolate DINOv2 position embeddings to avoid bicubic_aa ONNX ops."""
    import torch
    import torch.nn as nn

    emb = model.embeddings
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size)
        patches = emb.patch_embeddings(dummy)
        cls = emb.cls_token.expand(1, -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        pos = emb.interpolate_pos_encoding(tokens, img_size, img_size)
    emb.position_embeddings = nn.Parameter(pos.clone(), requires_grad=False)
    emb.interpolate_pos_encoding = lambda x, h, w: emb.position_embeddings
    return model


def _build_feature_module():
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    backbone = _bake_dinov2_pos_embed(
        AutoModel.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION).eval(),
        MODEL_INPUT_SIZE,
    )

    class _DinoFeatureExtractor(nn.Module):
        def __init__(self, inner: nn.Module):
            super().__init__()
            self.backbone = inner
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1))

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            # Map [-1, 1] input (from classify.py / verify-model.yml) to ImageNet stats.
            x = (pixel_values * 0.5 + 0.5 - self.mean) / self.std
            out = self.backbone(pixel_values=x, output_hidden_states=True)
            hs = out.hidden_states
            # Official DINOv2 linear-probe representation: last 4 CLS tokens + patch mean.
            return torch.cat(
                [
                    hs[-4][:, 0],
                    hs[-3][:, 0],
                    hs[-2][:, 0],
                    out.last_hidden_state[:, 0],
                    out.last_hidden_state[:, 5:].mean(dim=1),
                ],
                dim=-1,
            )

    return _DinoFeatureExtractor(backbone).eval()


def _quantize_weight_only(fp32_path: Path, output_path: Path, bits: int, block_size: int) -> None:
    """Apply weight-only block quantization so activations stay FP32 (zero batch drift)."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    quantizer = MatMulNBitsQuantizer(
        model=str(fp32_path),
        bits=bits,
        block_size=block_size,
        is_symmetric=True,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(str(output_path))


def cmd_extract(args: argparse.Namespace) -> None:
    import onnxruntime as ort
    import torch

    images, labels = load_cifar10_train(Path(args.data_dir))
    start, end = shard_slice(len(images), args.shard_index, args.num_shards)
    shard_indices = np.arange(start, end, dtype=np.int32)
    shard_images = images[start:end]
    shard_labels = labels[start:end]
    print(
        f"Extracting shard {args.shard_index}/{args.num_shards}: "
        f"indices [{start}, {end}) ({len(shard_indices)} images), flip={args.include_flip}",
        flush=True,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = out_path.with_suffix(".backbone_fp32.onnx")
    quant_path = out_path.with_suffix(".backbone_quant.onnx")
    try:
        module = _build_feature_module()
        torch.onnx.export(
            module,
            torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, dtype=torch.float32),
            str(fp32_path),
            input_names=["pixel_values"],
            output_names=["features"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "features": {0: "batch_size"}},
            opset_version=17,
            dynamo=False,
        )
        # Extract using the exact weight-only quantized backbone used at deployment so
        # training features match inference features identically.
        _quantize_weight_only(fp32_path, quant_path, args.quant_bits, args.quant_block_size)
        fp32_path.unlink(missing_ok=True)

        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads or (os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        session = ort.InferenceSession(
            str(quant_path), sess_options=options, providers=["CPUExecutionProvider"]
        )

        def _run_all(flip: bool) -> np.ndarray:
            parts: list[np.ndarray] = []
            total = len(shard_images)
            for offset in range(0, total, args.batch_size):
                batch = preprocess_batch(shard_images[offset : offset + args.batch_size], flip=flip)
                feats = session.run(["features"], {"pixel_values": batch})[0]
                parts.append(np.asarray(feats, dtype=np.float16))
                done = min(offset + args.batch_size, total)
                if done % 500 < args.batch_size or done == total:
                    tag = "flip" if flip else "orig"
                    print(f"[shard {args.shard_index} {tag}] {done}/{total}", flush=True)
            return np.concatenate(parts, axis=0)

        features = _run_all(flip=False)
        payload: dict[str, np.ndarray] = {
            "indices": shard_indices,
            "labels": shard_labels.astype(np.int64),
            "features": features,
        }
        if args.include_flip:
            payload["flip_features"] = _run_all(flip=True)
        np.savez(out_path, **payload)
        print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB, shape={features.shape})", flush=True)
    finally:
        fp32_path.unlink(missing_ok=True)
        quant_path.unlink(missing_ok=True)


def _load_all_shards(shards_dir: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    shard_files = sorted(shards_dir.rglob("*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"No .npz shard files found under {shards_dir}")
    idx_parts, feat_parts, flip_parts, label_parts = [], [], [], []
    has_flip = True
    for path in shard_files:
        with np.load(path, allow_pickle=False) as data:
            idx_parts.append(np.asarray(data["indices"], dtype=np.int64))
            feat_parts.append(np.asarray(data["features"], dtype=np.float32))
            label_parts.append(np.asarray(data["labels"], dtype=np.int64))
            if "flip_features" in data:
                flip_parts.append(np.asarray(data["flip_features"], dtype=np.float32))
            else:
                has_flip = False
    indices = np.concatenate(idx_parts, axis=0)
    order = np.argsort(indices)
    indices = indices[order]
    if not np.array_equal(indices, np.arange(50000, dtype=np.int64)):
        raise RuntimeError(
            f"Shards must cover indices 0..49999 exactly; got {len(indices)} entries "
            f"(min={indices.min()}, max={indices.max()})"
        )
    features = np.concatenate(feat_parts, axis=0)[order]
    labels = np.concatenate(label_parts, axis=0)[order]
    flip_features = np.concatenate(flip_parts, axis=0)[order] if has_flip and flip_parts else None
    return features, flip_features, labels


def _train_torch_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dim: int | None,
    weight_decay: float,
    lr: float,
    epochs: int,
    batch_size: int,
    label_smoothing: float,
    seed: int,
) -> tuple[float, object]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    in_dim = x_train.shape[1]
    if hidden_dim is None:
        head: nn.Module = nn.Linear(in_dim, 10)
    else:
        head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 10),
        )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for _epoch in range(epochs):
        head.train()
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(bx), by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        head.eval()
        with torch.no_grad():
            preds = head(x_val_t).argmax(dim=1).numpy()
        acc = float((preds == y_val).mean())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    return best_acc, head


def cmd_train(args: argparse.Namespace) -> None:
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler

    features, flip_features, labels = _load_all_shards(Path(args.shards_dir))
    train_idx, val_idx = stratified_split_indices(labels, val_size=args.val_size, seed=args.seed)

    x_train = features[train_idx]
    y_train = labels[train_idx]
    if flip_features is not None:
        x_train = np.concatenate([x_train, flip_features[train_idx]], axis=0)
        y_train = np.concatenate([y_train, labels[train_idx]], axis=0)
    x_val = features[val_idx]
    y_val = labels[val_idx]

    # Fit feature standardization strictly on the training split (never val or test).
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)

    candidates: list[dict[str, object]] = [
        {"name": "linear_wd1e-2", "hidden_dim": None, "weight_decay": 1e-2, "lr": 1e-3, "epochs": 25},
        {"name": "linear_wd5e-2", "hidden_dim": None, "weight_decay": 5e-2, "lr": 1e-3, "epochs": 25},
        {"name": "linear_wd2e-1", "hidden_dim": None, "weight_decay": 2e-1, "lr": 1e-3, "epochs": 25},
        {"name": "mlp512_wd5e-2", "hidden_dim": 512, "weight_decay": 5e-2, "lr": 5e-4, "epochs": 25},
        {"name": "mlp512_wd2e-1", "hidden_dim": 512, "weight_decay": 2e-1, "lr": 5e-4, "epochs": 25},
    ]

    val_results: dict[str, float] = {}
    best_name = ""
    best_val_acc = -1.0
    best_head: nn.Module | None = None
    for cfg in candidates:
        acc, head = _train_torch_head(
            x_train_s,
            y_train,
            x_val_s,
            y_val,
            hidden_dim=cfg["hidden_dim"],
            weight_decay=float(cfg["weight_decay"]),
            lr=float(cfg["lr"]),
            epochs=int(cfg["epochs"]),
            batch_size=256,
            label_smoothing=0.05,
            seed=args.seed,
        )
        name = str(cfg["name"])
        val_results[name] = round(acc, 6)
        print(f"[val] {name}: {acc:.4f} ({int(round(acc * len(y_val)))}/{len(y_val)})", flush=True)
        if acc > best_val_acc:
            best_val_acc = acc
            best_name = name
            best_head = head

    assert best_head is not None
    print(f"Selected head strictly on val split: {best_name} ({best_val_acc:.4f})", flush=True)

    # Fold StandardScaler into the first linear layer so ONNX needs no scaler op.
    first_linear = best_head if isinstance(best_head, nn.Linear) else best_head[0]
    w_folded, b_folded = fold_standard_scaler(
        first_linear.weight.detach().numpy(),
        first_linear.bias.detach().numpy(),
        scaler.mean_,
        scaler.scale_,
    )
    with torch.no_grad():
        first_linear.weight.copy_(torch.from_numpy(w_folded))
        first_linear.bias.copy_(torch.from_numpy(b_folded))

    # Sanity-check folded head against raw (unstandardized) val features.
    with torch.no_grad():
        folded_preds = best_head(torch.from_numpy(x_val)).argmax(dim=1).numpy()
    folded_acc = float((folded_preds == y_val).mean())
    print(f"Folded head val accuracy: {folded_acc:.4f}", flush=True)

    extractor = _build_feature_module()

    class _FullClassifier(nn.Module):
        def __init__(self, feat_mod: nn.Module, cls_head: nn.Module):
            super().__init__()
            self.feat_mod = feat_mod
            self.head = cls_head

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.head(self.feat_mod(pixel_values))

    full_model = _FullClassifier(extractor, best_head).eval()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fp32_path = output_path.with_suffix(".fp32.onnx")
    try:
        torch.onnx.export(
            full_model,
            torch.zeros(1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, dtype=torch.float32),
            str(fp32_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
            dynamo=False,
        )
        _quantize_weight_only(fp32_path, output_path, args.quant_bits, args.quant_block_size)
    finally:
        fp32_path.unlink(missing_ok=True)

    model_bytes = output_path.stat().st_size
    if model_bytes > MAX_MODEL_BYTES:
        raise RuntimeError(
            f"Exported ONNX model is {model_bytes / 1e6:.2f} MB (> {MAX_MODEL_BYTES / 1e6:.1f} MB limit)"
        )

    # Verify ONNX batch_size=1 parity on 200 validation images (still never touching test).
    images_raw, _ = load_cifar10_train(Path(args.data_dir))
    parity_indices = val_idx[:200]
    options = ort.SessionOptions()
    options.intra_op_num_threads = os.cpu_count() or 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    sess = ort.InferenceSession(str(output_path), sess_options=options, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    onnx_preds = []
    for idx in parity_indices:
        batch = preprocess_batch(images_raw[idx : idx + 1])
        logits = sess.run(None, {in_name: batch})[0]
        onnx_preds.append(int(np.argmax(logits[0])))
    onnx_preds_arr = np.asarray(onnx_preds, dtype=np.int64)
    onnx_val200_acc = float((onnx_preds_arr == labels[parity_indices]).mean())
    parity_agreement = float((onnx_preds_arr == folded_preds[:200]).mean())
    print(
        f"ONNX batch-1 val200 acc={onnx_val200_acc:.4f}, agreement_with_folded_head={parity_agreement:.4f}",
        flush=True,
    )

    digest = _sha256(output_path)
    run_id = os.environ.get("GITHUB_RUN_ID")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_url = (
        f"https://github.com/{repo}/actions/runs/{run_id}" if run_id and repo else None
    )
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/train_cifar10_dinov2.py",
        "model_file": output_path.name,
        "model_architecture": f"DINOv2-Reg ViT-B/14 + Trained Head (W{args.quant_bits} ONNX)",
        "model_sha256": digest,
        "model_bytes": model_bytes,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "base_model_license": BASE_MODEL_LICENSE,
        "pretraining_data": PRETRAINING_DATA,
        "dataset": "CIFAR-10",
        "train_images": int(len(train_idx)),
        "train_views": int(len(y_train)),
        "val_images": int(len(val_idx)),
        "test_set_used_for_training": False,
        "test_set_used_for_selection": False,
        "selected_head": best_name,
        "best_val_accuracy": round(folded_acc, 6),
        "val_candidates": val_results,
        "onnx_val200_accuracy": round(onnx_val200_acc, 6),
        "onnx_val200_agreement": round(parity_agreement, 6),
        "hyperparameters": {
            "seed": args.seed,
            "input_size": MODEL_INPUT_SIZE,
            "feature_dim": FEATURE_DIM,
            "flip_augmentation": flip_features is not None,
            "quantization": f"weight_only_int{args.quant_bits}_block{args.quant_block_size}_symmetric",
        },
        "github_run_id": run_id,
        "github_run_url": run_url,
    }
    prov_path = Path(args.provenance_out)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path} ({model_bytes / 1e6:.2f} MB, sha256={digest})", flush=True)
    print(f"wrote {prov_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="Extract DINOv2-Reg features for one training shard.")
    p_ext.add_argument("--data-dir", default=str(ROOT / "data"))
    p_ext.add_argument("--shard-index", type=int, required=True)
    p_ext.add_argument("--num-shards", type=int, required=True)
    p_ext.add_argument("--output", required=True)
    p_ext.add_argument("--batch-size", type=int, default=16)
    p_ext.add_argument("--threads", type=int, default=0)
    p_ext.add_argument("--include-flip", action="store_true")
    p_ext.add_argument("--quant-bits", type=int, default=8)
    p_ext.add_argument("--quant-block-size", type=int, default=128)

    p_tr = sub.add_parser("train", help="Train head on extracted shards and export ONNX.")
    p_tr.add_argument("--data-dir", default=str(ROOT / "data"))
    p_tr.add_argument("--shards-dir", required=True)
    p_tr.add_argument("--output", default=str(ROOT / MODEL_FILENAME))
    p_tr.add_argument(
        "--provenance-out",
        default=str(ROOT / "results" / "provenance" / "train_run.json"),
    )
    p_tr.add_argument("--val-size", type=int, default=5000)
    p_tr.add_argument("--seed", type=int, default=42)
    p_tr.add_argument("--quant-bits", type=int, default=8)
    p_tr.add_argument("--quant-block-size", type=int, default=128)

    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
