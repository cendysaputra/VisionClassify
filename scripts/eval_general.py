"""Evaluate General ONNX."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import general
from general_labels import LABELS

CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CATEGORY_EN = {
    "orang": "People",
    "bangunan": "Buildings",
    "alam": "Nature",
    "hewan": "Animals",
    "makanan": "Food",
    "kendaraan": "Vehicles",
    "benda": "Objects",
    "pakaian": "Clothing",
    "tempat": "Places",
}


def ensure_cifar100(root: Path) -> Path:
    folder = root / "cifar-100-python"
    if (folder / "test").is_file() and (folder / "meta").is_file():
        return folder
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-100-python.tar.gz"
    if not archive.is_file():
        urllib.request.urlretrieve(CIFAR100_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    return folder


def load_overlapping_test(root: Path, per_class: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load overlapping classes."""
    folder = ensure_cifar100(root)
    with (folder / "meta").open("rb") as handle:
        meta = pickle.load(handle, encoding="bytes")
    names = [item.decode() for item in meta[b"fine_label_names"]]
    key_to_idx = {label.key: index for index, label in enumerate(LABELS)}
    mapping = {ci: key_to_idx[name] for ci, name in enumerate(names) if name in key_to_idx}

    with (folder / "test").open("rb") as handle:
        entry = pickle.load(handle, encoding="bytes")
    raw_images = entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    raw_labels = np.asarray(entry[b"fine_labels"], dtype=np.int64)

    images, targets = [], []
    counts: dict[int, int] = {}
    for image, class_id in zip(raw_images, raw_labels):
        cid = int(class_id)
        if cid not in mapping or counts.get(cid, 0) >= per_class:
            continue
        counts[cid] = counts.get(cid, 0) + 1
        images.append(image)
        targets.append(mapping[cid])
    return np.asarray(images), np.asarray(targets, dtype=np.int64), sorted(mapping.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--per-class", type=int, default=30)
    # Batch size 1 like the app: the INT8 vision model quantizes per batch.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--embeddings", default=None,
                        help="Score another label embedding pack instead of the committed one.")
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()
    if args.embeddings and args.write_results:
        parser.error("--write-results only records the committed embedding pack")

    images, targets, subset_indices = load_overlapping_test(Path(args.data_dir), args.per_class)
    backend = general.load_backend()
    if args.embeddings:
        pack = general.load_embedding_pack(Path(args.embeddings))
        prompts = general._unit_rows(pack["embeddings"])
        label_index = pack["label_index"]
        print(f"embeddings: {args.embeddings}")
    else:
        prompts = general._unit_rows(backend.prompt_embeds)
        label_index = backend.prompt_labels
    subset_mask = np.zeros(len(LABELS), dtype=bool)
    subset_mask[subset_indices] = True

    top1_256 = 0
    top5_256 = 0
    top1_subset = 0
    cat_total: dict[str, int] = {}
    cat_top1: dict[str, int] = {}
    cat_top5: dict[str, int] = {}

    total = len(images)
    for start in range(0, total, args.batch_size):
        chunk = images[start : start + args.batch_size]
        batch = np.concatenate(
            [general.preprocess_general(Image.fromarray(img)) for img in chunk],
            axis=0,
        )
        # Run ONNX batch
        embs = general._unit_rows(
            backend._session.run(["image_embeds"], {backend._input_name: batch})[0]
        )
        for emb, target in zip(embs, targets[start : start + args.batch_size]):
            t = int(target)
            sims = prompts @ emb
            scores = np.full(len(LABELS), -np.inf, dtype=np.float32)
            np.maximum.at(scores, label_index, sims)
            order = np.argsort(-scores)
            cat = CATEGORY_EN[LABELS[t].category]
            cat_total[cat] = cat_total.get(cat, 0) + 1
            if int(order[0]) == t:
                top1_256 += 1
                cat_top1[cat] = cat_top1.get(cat, 0) + 1
            if t in order[:5]:
                top5_256 += 1
                cat_top5[cat] = cat_top5.get(cat, 0) + 1
            if int(np.argmax(np.where(subset_mask, scores, -np.inf))) == t:
                top1_subset += 1
        done = min(start + args.batch_size, total)
        if done % 320 < args.batch_size or done == total:
            print(f"[general] {done}/{total}", flush=True)

    per_category_top1 = {k: round(cat_top1.get(k, 0) / v, 4) for k, v in sorted(cat_total.items())}
    per_category_top5 = {k: round(cat_top5.get(k, 0) / v, 4) for k, v in sorted(cat_total.items())}
    print(f"51-way Top-1 : {top1_subset}/{total} = {top1_subset/total:.4f}")
    print(f"256-way Top-1: {top1_256}/{total} = {top1_256/total:.4f}")
    print(f"256-way Top-5: {top5_256}/{total} = {top5_256/total:.4f}")

    if not args.write_results:
        return

    payload = {
        "benchmark": "CIFAR-100 overlapping everyday classes (zero-shot)",
        "classes": len(subset_indices),
        "total_labels": len(LABELS),
        "images_evaluated": total,
        "batch_size": args.batch_size,
        "model_architecture": "CLIP ViT-B/16 (INT8 ONNX) + Prompt Ensemble",
        "model_sha256": general.VISION_SHA256,
        "model_bytes": general.VISION_BYTES,
        "subset_51_top1_accuracy": round(top1_subset / total, 4),
        "all_256_top1_accuracy": round(top1_256 / total, 4),
        "all_256_top5_accuracy": round(top5_256 / total, 4),
        "per_category_256_top1": per_category_top1,
        "per_category_256_top5": per_category_top5,
        "baseline_vit_b32": {
            "model_architecture": "CLIP ViT-B/32 (INT8 ONNX)",
            "model_bytes": 89117001,
            "subset_51_top1_accuracy": 0.7092,
            "all_256_top1_accuracy": 0.5503,
            "all_256_top5_accuracy": 0.817,
        },
    }
    out = ROOT / "results" / "general_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
