from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_cifar10  # noqa: E402


def test_stratified_split_is_balanced_disjoint_and_deterministic():
    labels = np.repeat(np.arange(10, dtype=np.int64), 50)
    train_a, val_a = train_cifar10.stratified_split_indices(labels, val_size=100, seed=42)
    train_b, val_b = train_cifar10.stratified_split_indices(labels, val_size=100, seed=42)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert len(train_a) == 400
    assert len(val_a) == 100
    assert set(train_a.tolist()).isdisjoint(set(val_a.tolist()))
    assert set(train_a.tolist()) | set(val_a.tolist()) == set(range(500))
    for cls in range(10):
        assert int((labels[val_a] == cls).sum()) == 10
        assert int((labels[train_a] == cls).sum()) == 40


def test_stratified_split_rejects_bad_val_size():
    labels = np.repeat(np.arange(10, dtype=np.int64), 5)
    with pytest.raises(ValueError):
        train_cifar10.stratified_split_indices(labels, val_size=0)
    with pytest.raises(ValueError):
        train_cifar10.stratified_split_indices(labels, val_size=15)
    with pytest.raises(ValueError):
        train_cifar10.stratified_split_indices(labels, val_size=50)


def test_train_batches_never_include_test_batch():
    assert train_cifar10.TRAIN_BATCH_FILES == (
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    )
    assert all("test" not in name for name in train_cifar10.TRAIN_BATCH_FILES)


def test_provenance_record_marks_test_split_untouched(tmp_path):
    model_file = tmp_path / "model_vit_cifar10.onnx"
    record = train_cifar10.build_provenance_record(
        output_path=model_file,
        model_sha256="abc123",
        model_bytes=1024,
        train_size=45000,
        val_size=5000,
        best_val_accuracy=0.986,
        best_epoch=4,
        epochs=5,
        batch_size=64,
        learning_rate=5e-5,
        weight_decay=0.01,
        seed=42,
        device="cuda",
    )
    assert record["base_model"] == train_cifar10.BASE_MODEL
    assert record["base_model_revision"] == train_cifar10.BASE_MODEL_REVISION
    assert record["base_model_license"] == "apache-2.0"
    assert record["test_set_used_for_training"] is False
    assert record["test_set_used_for_selection"] is False
    assert record["train_images"] == 45000
    assert record["val_images"] == 5000
