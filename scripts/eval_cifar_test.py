"""Evaluate CIFAR-10 ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classify import (
    CLASS_NAMES,
    MODEL_INPUT_SIZE,
    TTA_POLICIES,
    TTA_POLICY,
    load_cifar_backend,
    model_path,
    softmax,
    tta_views,
)

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def ensure_test_batch(root: Path) -> Path:
    folder = root / "cifar-10-batches-py"
    test_batch = folder / "test_batch"
    if test_batch.is_file():
        return test_batch
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    if not archive.is_file():
        print(f"Downloading CIFAR-10 from {CIFAR_URL}")
        urllib.request.urlretrieve(CIFAR_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    if not test_batch.is_file():
        raise FileNotFoundError(test_batch)
    return test_batch


def load_test(root: Path) -> tuple[np.ndarray, np.ndarray]:
    with ensure_test_batch(root).open("rb") as handle:
        entry = pickle.load(handle, encoding="bytes")
    images = entry[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(entry[b"labels"], dtype=np.int64)
    if images.shape != (10000, 32, 32, 3) or labels.shape != (10000,):
        raise RuntimeError(f"Unexpected test shape: {images.shape} {labels.shape}")
    return images, labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_batch(images_uint8: np.ndarray) -> np.ndarray:
    """Resize normalize batch."""
    batch = np.empty((len(images_uint8), 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.float32)
    for index, raw in enumerate(images_uint8):
        resized = Image.fromarray(raw).resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            Image.Resampling.BILINEAR,
        )
        array = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
        batch[index] = array.transpose(2, 0, 1)
    return batch


def predict(backend, images_uint8: np.ndarray, batch_size: int, policies: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Run batch evaluation."""
    predicted = {}
    total = len(images_uint8)
    for policy in policies:
        views = len(TTA_POLICIES[policy])
        # Cap batch views
        step = max(1, batch_size // views)
        classes = np.empty(total, dtype=np.int64)
        for start in range(0, total, step):
            chunk = preprocess_batch(images_uint8[start : start + step])
            logits = backend.predict_logits(tta_views(chunk, policy))
            probs = softmax(logits)
            averaged = probs.reshape(views, len(chunk), -1).mean(axis=0)
            classes[start : start + len(chunk)] = averaged.argmax(axis=1)
            done = min(start + step, total)
            if done % 2000 < step or done == total:
                print(f"[{policy}] {done}/{total}", flush=True)
        predicted[policy] = classes
    return predicted


def confusion(labels: np.ndarray, predicted: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    return matrix.tolist()


def per_class(matrix: list[list[int]]) -> dict[str, float]:
    scores = {}
    for index, name in enumerate(CLASS_NAMES):
        total = sum(matrix[index])
        scores[name] = matrix[index][index] / total
    return scores


def save_confusion_png(matrix: list[list[int]], path: Path) -> None:
    cell = 86
    left = 108
    top = 78
    width = left + 10 * cell + 20
    height = top + 10 * cell + 36
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((16, 16), f"CIFAR-10 test confusion matrix ({TTA_POLICY})", fill="black", font=font)
    draw.text((16, 36), "rows = true class, columns = predicted class, counts out of 1000", fill="black", font=font)
    peak = max(max(row) for row in matrix)
    for index, name in enumerate(CLASS_NAMES):
        draw.text((8, top + index * cell + 22), name, fill="black", font=font)
        draw.text((left + index * cell + 4, top - 18), name, fill="black", font=font)
    for row in range(10):
        for col in range(10):
            value = matrix[row][col]
            shade = int(255 - 220 * (value / peak))
            color = (shade, shade, 255) if row == col else (255, shade, shade)
            x0 = left + col * cell
            y0 = top + row * cell
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=color, outline="white")
            draw.text((x0 + 8, y0 + 28), str(value), fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(model_path()))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    # The INT8 model quantizes activations per batch, so a prediction depends on
    # the other images in the batch. The app scores one image at a time; batch
    # size 1 measures what the app does.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--policies", nargs="+", default=[TTA_POLICY])
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()

    path = Path(args.model)
    policies = tuple(dict.fromkeys(args.policies))
    backend = load_cifar_backend(path)
    images, labels = load_test(Path(args.data_dir))
    predicted = predict(backend, images, args.batch_size, policies)
    correct = {policy: int((labels == value).sum()) for policy, value in predicted.items()}
    matrices = {policy: confusion(labels, value) for policy, value in predicted.items()}
    digest = sha256(path)

    print(f"model: {path.name}")
    print(f"sha256: {digest}")
    for policy in policies:
        marker = " (app)" if policy == TTA_POLICY else ""
        print(f"test {policy}{marker}: {correct[policy]}/10000 = {correct[policy] / 10000:.4f}")
    app_policy = TTA_POLICY if TTA_POLICY in matrices else policies[0]
    app_class = per_class(matrices[app_policy])
    print(f"per class ({app_policy}):")
    for name in CLASS_NAMES:
        print(f"  {name}: {app_class[name]:.4f}")

    if not args.write_results:
        return

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    train_prov_path = results / "provenance" / "train_run.json"
    train_prov = None
    if train_prov_path.is_file():
        candidate = json.loads(train_prov_path.read_text(encoding="utf-8"))
        if candidate.get("model_sha256") == digest:
            train_prov = candidate

    if train_prov is not None:
        model_architecture = str(train_prov.get("model_architecture", "ViT (ONNX)"))
        source_model = f"{train_prov['script']} (base: {train_prov['base_model']})"
        test_for_training = train_prov.get("test_set_used_for_training")
        test_for_selection = train_prov.get("test_set_used_for_selection")
        provenance_ref = "results/provenance/train_run.json"
    else:
        model_architecture = "ViT-Small/16 (INT8 ONNX)"
        source_model = "MF21377197/vit-small-patch16-224-finetuned-Cifar10"
        # Third-party weights: whether the test split was used during their
        # fine-tune is not known from this repo. See results/provenance/.
        test_for_training = None
        test_for_selection = None
        provenance_ref = "results/provenance/"

    payload = {
        "dataset": "CIFAR-10",
        "test_images": 10000,
        "model_file": path.name,
        "model_architecture": model_architecture,
        "source_model": source_model,
        "model_sha256": digest,
        "model_bytes": path.stat().st_size,
        "app_inference": app_policy,
        "app_views": len(TTA_POLICIES[app_policy]),
        "test_set_used_for_training": test_for_training,
        "test_set_used_for_selection": test_for_selection,
        "provenance": provenance_ref,
        "batch_size": args.batch_size,
        "test_correct": correct,
        "test_accuracy": {policy: count / 10000 for policy, count in correct.items()},
        "per_class_test_accuracy": app_class,
        "confusion_matrix": matrices[app_policy],
        "confusion_matrix_png": "results/confusion_matrix_tta.png",
    }
    metrics_path = results / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (results / "model_sha256.txt").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    save_confusion_png(matrices[app_policy], results / "confusion_matrix_tta.png")
    print(f"wrote {metrics_path}")
    print(f"wrote {results / 'model_sha256.txt'}")
    print(f"wrote {results / 'confusion_matrix_tta.png'}")


if __name__ == "__main__":
    main()
