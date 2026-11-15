"""Skor model CIFAR-10 yang sudah dibekukan pada 10.000 gambar uji resmi.

Himpunan uji tidak dipakai untuk melatih atau memilih model. Skrip ini hanya
mengukur ulang berkas model dengan tampilan yang sama seperti aplikasi
(classify.TTA_POLICY). Angka yang dicetak harus sama dengan results/metrics.json.

Butuh dependensi aplikasi (TensorFlow), bukan PyTorch:

    python scripts/eval_cifar_test.py
    python scripts/eval_cifar_test.py --write-results
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classify import CLASS_NAMES, TTA_POLICIES, TTA_POLICY, model_path, tta_views

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
# Tampilan yang dilaporkan: satu gambar, cermin, dan kebijakan aplikasi.
REPORTED_POLICIES = tuple(dict.fromkeys(("single", "flip", TTA_POLICY)))

# Fakta pelatihan model dasar v5 yang tidak bisa dibaca dari berkas .h5.
# Model yang dipakai aplikasi adalah lanjutan SWA dari v5; faktanya ada di
# results/swa_metrics.json, yang ditulis scripts/finetune_swa.py.
BASE_TRAINING_RECORD = {
    "model_file": "model_cnn_cifar10_v5.h5",
    "model_sha256": "539a2abe2e9145eeaff34e21a295d77bf12c4db5f0967143db98467b08d52b71",
    "train_images": 45000,
    "validation_images": 5000,
    "validation_per_class": 500,
    "seed": 42,
    "epochs_completed": 200,
    "best_checkpoint_epoch": 196,
    "checkpoint_metric": "validation_accuracy_horizontal_flip_tta",
    "validation_accuracy_single_view": 0.9624,
    "validation_accuracy_flip_tta": 0.9648,
    "test_correct_single_view": 9571,
    "test_correct_flip_tta": 9606,
    "training_time_seconds": 23566.9,
    "date": "2026-11-08",
    "framework_train": "PyTorch 2.15.0+cpu",
}


def ensure_test_batch(root: Path) -> Path:
    folder = root / "cifar-10-batches-py"
    test_batch = folder / "test_batch"
    if test_batch.is_file():
        return test_batch
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "cifar-10-python.tar.gz"
    if not archive.is_file():
        print(f"Mengunduh CIFAR-10 dari {CIFAR_URL}")
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
        raise RuntimeError(f"Bentuk uji tidak terduga: {images.shape} {labels.shape}")
    return images, labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(model, images_uint8: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    """Kelas tebakan per kebijakan tampilan, lewat classify.tta_views seperti aplikasi."""
    predicted = {}
    for policy in REPORTED_POLICIES:
        views = len(TTA_POLICIES[policy])
        # batch_size membatasi jumlah tampilan per panggilan model, bukan jumlah gambar.
        step = max(1, batch_size // views)
        classes = np.empty(len(images_uint8), dtype=np.int64)
        for start in range(0, len(images_uint8), step):
            batch = images_uint8[start : start + step].astype(np.float32) / 255.0
            probs = np.asarray(model(tta_views(batch, policy), training=False), dtype=np.float64)
            averaged = probs.reshape(views, len(batch), -1).mean(axis=0)
            classes[start : start + len(batch)] = averaged.argmax(axis=1)
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
    draw.text((16, 16), f"CIFAR-10 test confusion matrix, app views ({TTA_POLICY})", fill="black", font=font)
    draw.text((16, 36), "rows = true class, columns = predicted class, counts out of 1000", fill="black", font=font)
    peak = max(max(row) for row in matrix)
    short = list(CLASS_NAMES)
    for index, name in enumerate(short):
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
            label = str(value)
            draw.text((x0 + 8, y0 + 28), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(model_path()))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()

    path = Path(args.model)
    from tensorflow.keras.models import load_model

    model = load_model(path, compile=False)
    images, labels = load_test(Path(args.data_dir))
    predicted = predict(model, images, args.batch_size)
    correct = {policy: int((labels == value).sum()) for policy, value in predicted.items()}
    matrices = {policy: confusion(labels, value) for policy, value in predicted.items()}
    digest = sha256(path)

    print(f"model: {path.name}")
    print(f"sha256: {digest}")
    for policy in REPORTED_POLICIES:
        marker = " (aplikasi)" if policy == TTA_POLICY else ""
        print(f"test {policy}{marker}: {correct[policy]}/10000 = {correct[policy] / 10000:.4f}")
    print(f"per class single / {TTA_POLICY}:")
    single_class = per_class(matrices["single"])
    app_class = per_class(matrices[TTA_POLICY])
    for name in CLASS_NAMES:
        print(f"  {name}: {single_class[name]:.4f} / {app_class[name]:.4f}")

    if not args.write_results:
        return

    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    finetune = {}
    swa_path = results / "swa_metrics.json"
    if swa_path.is_file():
        finetune = json.loads(swa_path.read_text(encoding="utf-8"))
        finetune.pop("history", None)
    payload = {
        "dataset": "CIFAR-10",
        "test_images": 10000,
        "model_file": path.name,
        "model_sha256": digest,
        "model_bytes": path.stat().st_size,
        "app_inference": TTA_POLICY,
        "app_views": len(TTA_POLICIES[TTA_POLICY]),
        "test_set_used_for_training": False,
        "test_set_used_for_selection": False,
        "test_correct": correct,
        "test_accuracy": {policy: count / 10000 for policy, count in correct.items()},
        "per_class_test_accuracy_single": single_class,
        "per_class_test_accuracy_app": app_class,
        "confusion_matrix_single": matrices["single"],
        "confusion_matrix_app": matrices[TTA_POLICY],
        "confusion_matrix_png": "results/confusion_matrix_tta.png",
        "fine_tune": finetune,
        "base_training": BASE_TRAINING_RECORD,
    }
    metrics_path = results / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (results / "model_sha256.txt").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    save_confusion_png(matrices[TTA_POLICY], results / "confusion_matrix_tta.png")
    print(f"wrote {metrics_path}")
    print(f"wrote {results / 'model_sha256.txt'}")
    print(f"wrote {results / 'confusion_matrix_tta.png'}")


if __name__ == "__main__":
    main()
