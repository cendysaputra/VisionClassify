from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_cifar10_dinov2 as trainer  # noqa: E402


def test_shard_slice_partitions_total_without_gaps_or_overlap():
    total = 50000
    num_shards = 20
    covered = []
    for idx in range(num_shards):
        start, end = trainer.shard_slice(total, idx, num_shards)
        assert start < end
        covered.extend(range(start, end))
    assert covered == list(range(total))


def test_shard_slice_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        trainer.shard_slice(100, -1, 4)
    with pytest.raises(ValueError):
        trainer.shard_slice(100, 4, 4)
    with pytest.raises(ValueError):
        trainer.shard_slice(100, 0, 0)


def test_fold_standard_scaler_matches_separate_transform():
    rng = np.random.default_rng(7)
    x = rng.normal(loc=3.0, scale=2.5, size=(16, 12)).astype(np.float32)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    weight = rng.normal(size=(10, 12)).astype(np.float32)
    bias = rng.normal(size=(10,)).astype(np.float32)

    expected = ((x - mean) / scale) @ weight.T + bias
    w_folded, b_folded = trainer.fold_standard_scaler(weight, bias, mean, scale)
    actual = x @ w_folded.T + b_folded
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_preprocess_batch_matches_classify_preprocess():
    from PIL import Image
    from classify import preprocess

    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
    batched = trainer.preprocess_batch(raw)
    for i in range(2):
        np.testing.assert_array_equal(batched[i : i + 1], preprocess(Image.fromarray(raw[i])))
