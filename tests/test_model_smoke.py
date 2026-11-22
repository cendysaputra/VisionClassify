import pytest
from PIL import Image

from classify import CLASS_NAMES, load_cifar_backend, model_path


def test_saved_model_outputs_a_probability_distribution():
    pytest.importorskip("onnxruntime")

    path = model_path()
    if not path.is_file():
        pytest.skip(f"{path.name} missing")

    backend = load_cifar_backend(path)
    image = Image.new("RGB", (48, 36), (30, 120, 200))
    summary = backend.predict(image)

    assert summary.label in CLASS_NAMES
    total = sum(item.probability for item in summary.ranking)
    assert total == pytest.approx(1.0, abs=1e-4)
    assert all(item.probability >= -1e-5 for item in summary.ranking)
    assert len(summary.ranking) == len(CLASS_NAMES)
