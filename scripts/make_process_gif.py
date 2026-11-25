"""Animate the CIFAR-10 prediction process on one real test image (README figure).

Not part of the Streamlit runtime. Needs matplotlib on top of requirements.txt:

    pip install -r requirements.txt matplotlib
    python scripts/make_process_gif.py --index 0 --out docs/process.gif

Everything drawn comes from the committed model and the app code: classify.preprocess
makes the 224x224 input, the ONNX model gives the top-5 bars, and the heat map comes
from the same model.

Heat map: the model sees only one 56x56 window of the input, the rest set to mid grey,
and we record its probability for the top-1 class; each 28 px cell shows the mean over
the windows that cover it. The usual occlusion map (grey out one square, measure the
drop) is flat for this model: with half of a test photo greyed out its top-1
probability moved by under one point. The model exposes no attention weights, so a
window map is the explanation it can give without changing the export.

The image is test_batch[--index]. Pick the index before looking at predictions, so
the figure is not a hand-picked success; the default is the first test image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from classify import (  # noqa: E402
    CLASS_NAMES,
    MODEL_INPUT_SIZE,
    center_square,
    load_cifar_backend,
    model_path,
    preprocess,
    softmax,
)
from eval_cifar_test import load_test  # noqa: E402

INK, INK2, MUTED, GRID = "#191919", "#666666", "#bdbdbd", "#e6e6e6"
ACCENT = "#257de0"
STEPS = ["1  Test image", "2  Resize 224×224", "3  Model + window map", "4  Prediction"]
# Frames per phase: pixel reveal, upscale fade, window sweep, bars grow, hold.
PHASES = [18, 16, 30, 14, 26]
WINDOW, STRIDE = 56, 28


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_name(path: Path, digest: str) -> str:
    """Architecture from the training record, only if it describes this exact file."""
    record = ROOT / "results" / "provenance" / "train_run.json"
    if record.is_file():
        info = json.loads(record.read_text())
        if info.get("model_sha256") == digest and info.get("model_architecture"):
            return str(info["model_architecture"])
    return path.name


def window_scores(backend, batch: np.ndarray, target: int) -> list[tuple[int, int, float]]:
    """(top, left, P(target)) with only one WINDOW x WINDOW square visible, the rest 0 (mid grey).

    One image per call, as in the app: the model quantizes per batch, so batching the
    windows would change their scores.
    """
    scores = []
    for top in range(0, MODEL_INPUT_SIZE - WINDOW + 1, STRIDE):
        for left in range(0, MODEL_INPUT_SIZE - WINDOW + 1, STRIDE):
            masked = np.zeros_like(batch)
            region = (slice(None), slice(None), slice(top, top + WINDOW), slice(left, left + WINDOW))
            masked[region] = batch[region]
            scores.append((top, left, float(softmax(backend.predict_logits(masked))[0, target])))
    return scores


def cell_map(scores: list[tuple[int, int, float]]) -> np.ndarray:
    """Mean window score per STRIDE-sized cell; NaN where no window has been scored yet."""
    cells = MODEL_INPUT_SIZE // STRIDE
    total = np.zeros((cells, cells))
    count = np.zeros((cells, cells))
    for top, left, score in scores:
        rows = slice(top // STRIDE, (top + WINDOW) // STRIDE)
        cols = slice(left // STRIDE, (left + WINDOW) // STRIDE)
        total[rows, cols] += score
        count[rows, cols] += 1
    with np.errstate(invalid="ignore"):
        return total / count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=0, help="CIFAR-10 test_batch index")
    ap.add_argument("--image", help="use this image file instead of a CIFAR-10 test image (local preview)")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "process.gif"))
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    if args.image:
        raw = Image.open(args.image).convert("RGB")
        true_label = None
        source = Path(args.image).name
    else:
        images, labels = load_test(Path(args.data))
        raw = Image.fromarray(images[args.index])
        true_label = CLASS_NAMES[int(labels[args.index])]
        source = f"CIFAR-10 test image #{args.index}"

    path = model_path()
    digest = sha256(path)
    backend = load_cifar_backend(path)
    batch = preprocess(raw)
    probs = softmax(backend.predict_logits(batch))[0]
    top = np.argsort(probs)[::-1][:5]
    names = [CLASS_NAMES[i] for i in top]
    scores = window_scores(backend, batch, int(top[0]))
    full = cell_map(scores)
    lo, hi = float(np.nanmin(full)), float(np.nanmax(full))
    print("window map, P(%s) per cell:\n%s" % (names[0], np.round(full, 2)))

    small = np.asarray(center_square(raw).resize((32, 32), Image.Resampling.BILINEAR))
    # Show exactly what the model sees: undo the [-1, 1] scaling of classify.preprocess.
    big = ((batch[0].transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    cmap = plt.get_cmap("inferno")

    def overlay(values: np.ndarray, fade: float = 1.0) -> np.ndarray:
        scaled = np.nan_to_num((values - lo) / (hi - lo) if hi > lo else values * 0, nan=0.0)
        rgba = cmap(np.kron(scaled, np.ones((STRIDE, STRIDE))))
        seen = np.kron(~np.isnan(values), np.ones((STRIDE, STRIDE)))
        rgba[..., 3] = 0.7 * seen * fade
        return rgba

    fig = plt.figure(figsize=(10, 5.4), dpi=100)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.1,
                          left=0.04, right=0.97, top=0.8, bottom=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[0, 2])

    step_txt = [fig.text(0.04 + i * 0.24, 0.93, s, fontsize=11, color=MUTED, weight="bold")
                for i, s in enumerate(STEPS)]
    truth = f"  ·  true label: {true_label}" if true_label else ""
    fig.text(0.04, 0.88, f"{source}{truth}  ·  {model_name(path, digest)}  ·  sha256 {digest[:12]}",
             fontsize=8.5, color=INK2)
    note = fig.text(0.04 + 0.335, 0.05, "", fontsize=8, color=INK2)

    blank_small = np.full_like(small, 255)
    im_a = ax_a.imshow(blank_small, interpolation="nearest")
    ax_a.set_title("32×32 input", fontsize=9, color=INK2, loc="left")
    im_b = ax_b.imshow(np.full_like(big, 255))
    cam_im = ax_b.imshow(overlay(np.full_like(full, np.nan)))
    ax_b.set_title("224×224 model input", fontsize=9, color=INK2, loc="left")
    box = plt.Rectangle((0, 0), WINDOW, WINDOW, fill=False, edgecolor="white", lw=1.5, visible=False)
    ax_b.add_patch(box)
    for a in (ax_a, ax_b):
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_color(GRID)

    bars = ax_p.barh(range(5), [0] * 5, color=[ACCENT] + [MUTED] * 4, height=0.55)
    ax_p.set_xlim(0, 1.3)
    ax_p.set_ylim(4.6, -0.7)
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    ax_p.set_title("top-5 classes", fontsize=9, color=INK2, loc="left")
    labels = [ax_p.text(0.0, i - 0.36, n, fontsize=9, color=INK, va="bottom", alpha=0) for i, n in enumerate(names)]
    values = [ax_p.text(0, i, "", fontsize=9, color=INK, va="center") for i in range(5)]
    for s in ax_p.spines.values():
        s.set_visible(False)

    bounds = np.cumsum([0] + PHASES)

    def set_step(k: int) -> None:
        for i, t in enumerate(step_txt):
            t.set_color(ACCENT if i == k else (INK if i < k else MUTED))

    def update(f: int):
        if f < bounds[1]:  # reveal the 32x32 image row by row
            set_step(0)
            rows = int(32 * (f + 1) / PHASES[0])
            shown = blank_small.copy()
            shown[:rows] = small[:rows]
            im_a.set_data(shown)
        elif f < bounds[2]:  # upscaled input fades in
            set_step(1)
            im_a.set_data(small)
            g = (f - bounds[1] + 1) / PHASES[1]
            im_b.set_data((255 * (1 - g) + big * g).astype(np.uint8))
        elif f < bounds[3]:  # window sweeps, heat map fills in behind it
            set_step(2)
            im_b.set_data(big)
            done = int(len(scores) * (f - bounds[2] + 1) / PHASES[2])
            cam_im.set_data(overlay(cell_map(scores[:done])))
            if done < len(scores):
                wy, wx, _ = scores[done]
                box.set_xy((wx - 0.5, wy - 0.5))
                box.set_visible(True)
            else:
                box.set_visible(False)
            note.set_text(f"brighter = higher P('{names[0]}') when the model sees only that {WINDOW}×{WINDOW} window "
                          f"(P {lo:.2f} to {hi:.2f})")
        else:  # bars grow, then hold
            set_step(3)
            box.set_visible(False)
            g = min(1.0, (f - bounds[3] + 1) / PHASES[3])
            for i, (b, idx) in enumerate(zip(bars, top)):
                w = float(probs[idx]) * g
                b.set_width(w)
                values[i].set_position((w + 0.02, i))
                values[i].set_text(f"{100 * probs[idx] * g:.1f}%")
                labels[i].set_alpha(1)
        return []

    anim = FuncAnimation(fig, update, frames=int(bounds[-1]), blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {out}: {int(bounds[-1])} frames, top-1 {names[0]} {100 * probs[top[0]]:.1f}%"
          + (f", true {true_label}" if true_label else ""))


if __name__ == "__main__":
    main()
