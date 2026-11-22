"""CIFAR-10 ONNX classifier."""

from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

CLASS_NAMES: tuple[str, ...] = (
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
)

MODEL_FILENAME = "model_vit_cifar10.onnx"
MODEL_INPUT_SIZE = 224
PREVIEW_MAX_SIDE = 512
# Limit image pixels
MAX_IMAGE_PIXELS = 40_000_000
LOW_CONFIDENCE_THRESHOLD = 0.5
TTA_POLICIES: dict[str, tuple[bool, ...]] = {
    "single": (False,),
    "flip": (False, True),
}
# Active TTA policy
TTA_POLICY = "single"

_UNREADABLE = "Gambar tidak dapat dibaca. Gunakan file JPG, PNG, atau WEBP yang valid."


class ImageLoadError(ValueError):
    """Unreadable image error."""


@dataclass(frozen=True)
class RankedClass:
    label: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    low_confidence: bool
    ranking: tuple[RankedClass, ...]


def model_path(base_dir: Path | None = None) -> Path:
    root = Path(__file__).resolve().parent if base_dir is None else Path(base_dir)
    return root / MODEL_FILENAME


def load_image(source: bytes | bytearray | io.BufferedIOBase) -> Image.Image:
    """Decode RGB image."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(source) as opened:
            opened.load()
            transposed = ImageOps.exif_transpose(opened) or opened
            rgb = transposed.convert("RGB")
            image = rgb.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as exc:
        raise ImageLoadError(_UNREADABLE) from exc

    if image.width < 1 or image.height < 1:
        raise ImageLoadError(_UNREADABLE)
    return image


def preview_image(image: Image.Image, max_side: int = PREVIEW_MAX_SIDE) -> Image.Image:
    """Resize preview image."""
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    return preview


def center_square(image: Image.Image) -> Image.Image:
    """Center square crop."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    # Detach cropped pixels
    return image.crop((left, top, left + side, top + side)).copy()


def preprocess(image: Image.Image) -> np.ndarray:
    """Preprocess ViT batch."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    prepared = center_square(image)
    if max(prepared.size) > PREVIEW_MAX_SIDE:
        prepared.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), Image.Resampling.BILINEAR)
    resized = prepared.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)
    array = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
    if array.shape != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3):
        raise ImageLoadError("Gambar tidak dapat diubah ke ukuran masukan model.")
    nchw = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(nchw[np.newaxis, ...], dtype=np.float32)


def horizontal_flip(batch: np.ndarray) -> np.ndarray:
    """Flip NCHW width."""
    array = np.asarray(batch)
    if array.ndim != 4:
        raise ValueError("Batch cermin harus berbentuk (N, kanal, tinggi, lebar).")
    return np.ascontiguousarray(np.flip(array, axis=3))


def tta_views(batch: np.ndarray, policy: str = TTA_POLICY) -> np.ndarray:
    """Build TTA views."""
    if policy not in TTA_POLICIES:
        raise ValueError(f"Kebijakan TTA tidak dikenal: {policy!r}")
    views = [horizontal_flip(batch) if flip else np.asarray(batch) for flip in TTA_POLICIES[policy]]
    return np.concatenate(views, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute stable softmax."""
    values = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Model mengembalikan nilai yang tidak valid.")
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / np.sum(exponents, axis=-1, keepdims=True)


def average_probabilities(views: np.ndarray) -> np.ndarray:
    """Average view probabilities."""
    rows = np.asarray(views, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Rata-rata membutuhkan bentuk (tampilan, {len(CLASS_NAMES)}), "
            f"diterima {tuple(np.asarray(views).shape)}."
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError("Probabilitas berisi nilai yang tidak valid.")
    return rows.mean(axis=0)


def class_probabilities(output: np.ndarray) -> np.ndarray:
    """Validate probability vector."""
    probs = np.asarray(output, dtype=np.float64)
    expected = (len(CLASS_NAMES),)
    if probs.shape == (1, len(CLASS_NAMES)):
        probs = probs[0]
    if probs.shape != expected:
        raise ValueError(
            f"Model mengembalikan bentuk {tuple(np.asarray(output).shape)}, "
            f"diharapkan {expected} atau (1, {len(CLASS_NAMES)})."
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError("Model mengembalikan nilai yang tidak valid.")
    return probs


def top_indices(probabilities: np.ndarray, k: int = 3) -> list[int]:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size == 0 or k <= 0:
        return []
    order = np.argsort(probs)[::-1]
    return [int(index) for index in order[: min(k, probs.size)]]


def summarize_prediction(output: np.ndarray) -> Prediction:
    probs = class_probabilities(output)
    ranking = tuple(
        RankedClass(CLASS_NAMES[index], float(probs[index]))
        for index in top_indices(probs, len(CLASS_NAMES))
    )
    top = ranking[0]
    return Prediction(
        label=top.label,
        confidence=top.probability,
        low_confidence=top.probability < LOW_CONFIDENCE_THRESHOLD,
        ranking=ranking,
    )


class CifarBackend:
    """ONNX CIFAR-10 backend."""

    def __init__(self, session: object):
        self._session = session
        self._lock = threading.Lock()
        self._input_name = session.get_inputs()[0].name
        self._output_name = session.get_outputs()[0].name

    def predict_logits(self, batch: np.ndarray) -> np.ndarray:
        with self._lock:
            output = self._session.run([self._output_name], {self._input_name: batch})[0]
        return np.asarray(output, dtype=np.float32)

    def predict(self, image: Image.Image, policy: str = TTA_POLICY) -> Prediction:
        batch = preprocess(image)
        logits = self.predict_logits(tta_views(batch, policy))
        probs = softmax(logits)
        return summarize_prediction(average_probabilities(probs))


def load_cifar_backend(path: Path | None = None) -> CifarBackend:
    """Load ONNX session."""
    source = model_path() if path is None else Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(source),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return CifarBackend(session)
