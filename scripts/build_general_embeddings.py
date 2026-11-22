"""Rebuild General mode label embeddings.

Not part of the Streamlit runtime. Needs onnxruntime, tokenizers and numpy:

    pip install onnxruntime tokenizers numpy
    python scripts/build_general_embeddings.py

The CLIP ViT-B/16 text encoder and tokenizer are downloaded from a pinned
Hugging Face revision (the same revision as the vision encoder in general.py),
so the output only changes when general_labels.py or this script changes.

One embedding row is written per prompt in general_labels.prompt_texts(), so
the row layout always matches the code. With --templates ensemble, a label
without hand-written prompts gets the mean of several template sentences in
its single row (prompt ensembling, Radford et al. 2021). Hand-written prompts
are embedded as written. The committed pack uses --templates single: on the
benchmark in scripts/eval_general.py it beat both the earlier ensembled pack
and this ensemble on every metric (results/general_metrics.json).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from general_labels import LABELS, NULL_PROMPTS, GeneralLabel, prompt_texts  # noqa: E402

# Same repository and revision as general.VISION_URL.
_REVISION_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/"
    "342fdf2f67aded64d138ff074745fb4a5d2bba5f/"
)
TEXT_URL = _REVISION_URL + "onnx/text_model_quantized.onnx"
TOKENIZER_URL = _REVISION_URL + "tokenizer.json"
CONTEXT_LENGTH = 77
PAD_ID = 49407
EMBED_DIM = 512
OUTPUT = ROOT / "data" / "general_label_embeddings.npz"

# Templates for labels without hand-written prompts. {a} is "a" or "an".
ENSEMBLE_TEMPLATES: tuple[str, ...] = (
    "a photo of {a} {name}.",
    "a photo of the {name}.",
    "a close-up photo of {a} {name}.",
    "a bright photo of {a} {name}.",
    "a cropped photo of {a} {name}.",
    "a good photo of {a} {name}.",
    "a low resolution photo of {a} {name}.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_file(env_name: str, url: str, filename: str) -> Path:
    """Download once into the cache. The URL pins a revision, so content is fixed."""
    override = os.environ.get(env_name)
    if override:
        return Path(override)
    cache = Path(os.environ.get("GENERAL_MODEL_CACHE", Path.home() / ".cache" / "VisionClassify"))
    path = cache / filename
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "VisionClassify/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temporary.replace(path)
    return path


def _article(name: str) -> str:
    return "an" if name[:1].lower() in "aeiou" else "a"


def row_sentences(label: GeneralLabel, templates: str) -> list[tuple[str, ...]]:
    """Sentences averaged into each embedding row, one group per prompt_texts() row."""
    rows = prompt_texts(label)
    if label.prompts or templates == "single":
        return [(text,) for text in rows]
    article = _article(label.name_en)
    return [tuple(t.format(a=article, name=label.name_en) for t in ENSEMBLE_TEMPLATES)]


def _encode(tokenizer: object, prompts: list[str]) -> np.ndarray:
    rows = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        if len(ids) != CONTEXT_LENGTH:
            raise SystemExit(f"Token length is not {CONTEXT_LENGTH}: {prompt}")
        rows.append(ids)
    return np.asarray(rows, dtype=np.int64)


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms < 1e-6):
        raise SystemExit("Empty text embedding")
    return vectors / norms


def _embed(session: object, tokenizer: object, prompts: list[str], batch: int = 64) -> np.ndarray:
    parts = []
    for start in range(0, len(prompts), batch):
        ids = _encode(tokenizer, prompts[start : start + batch])
        output = session.run(["text_embeds"], {"input_ids": ids})[0]
        parts.append(np.asarray(output, dtype=np.float32))
    return _unit(np.concatenate(parts, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", choices=("single", "ensemble"), default="single")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    import onnxruntime as ort
    from tokenizers import Tokenizer

    text_path = _cache_file("CLIP_TEXT_ONNX", TEXT_URL, "clip-vit-b16-text-int8.onnx")
    tokenizer_path = _cache_file("CLIP_TOKENIZER", TOKENIZER_URL, "clip-vit-b16-tokenizer.json")
    print(f"text model sha256: {_sha256(text_path)}")
    print(f"tokenizer sha256: {_sha256(tokenizer_path)}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=CONTEXT_LENGTH)
    tokenizer.enable_padding(length=CONTEXT_LENGTH, pad_id=PAD_ID)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(text_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    groups: list[tuple[str, ...]] = []
    label_index: list[int] = []
    for index, label in enumerate(LABELS):
        for group in row_sentences(label, args.templates):
            groups.append(group)
            label_index.append(index)
    flat = [sentence for group in groups for sentence in group]
    flat_embeds = _embed(session, tokenizer, flat)
    rows, start = [], 0
    for group in groups:
        rows.append(flat_embeds[start : start + len(group)].mean(axis=0))
        start += len(group)
    embeddings = _unit(np.stack(rows)).astype(np.float32)
    null_embeddings = _embed(session, tokenizer, list(NULL_PROMPTS))
    if embeddings.shape[1] != EMBED_DIM:
        raise SystemExit(f"Unexpected embedding size {embeddings.shape[1]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        embeddings=embeddings,
        label_index=np.asarray(label_index, dtype=np.int32),
        null_embeddings=null_embeddings,
        keys=np.asarray([label.key for label in LABELS]),
    )
    print(
        f"wrote {output} ({embeddings.shape[0]} rows from {len(flat)} sentences, "
        f"{len(LABELS)} labels, templates={args.templates})"
    )


if __name__ == "__main__":
    main()
