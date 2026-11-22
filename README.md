<div align="center">

# Image Classification App

Lightweight Streamlit and ONNX Runtime web app achieving **98.42%** CIFAR 10 test accuracy plus **256** general object labels.

![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B)
![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX%20INT8-005CED)
![Test Accuracy](https://img.shields.io/badge/Accuracy-98.42%25-2EA043)
![Model Size](https://img.shields.io/badge/Model-22.4%20MB-6F42C1)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

## Screenshots

<table>
  <tr>
    <td align="center" width="50%">
      <img src="results/screenshot_cifar10.png" alt="CIFAR 10 Mode" width="100%" /><br />
      <sub><b>CIFAR 10 Mode</b>: ViT Small INT8 ONNX (10 classes)</sub>
    </td>
    <td align="center" width="50%">
      <img src="results/screenshot_umum.png" alt="General Mode" width="100%" /><br />
      <sub><b>General Mode</b>: CLIP ViT B/32 ONNX (256 labels)</sub>
    </td>
  </tr>
</table>

## Performance

![Model Accuracy and Size Overview](results/accuracy_overview.png)

<table>
  <thead>
    <tr>
      <th align="left">Metric</th>
      <th align="center">v6</th>
      <th align="center">v7 (Current)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Model</td>
      <td align="center">ResNet 18 (32×32)</td>
      <td align="center"><b>ViT Small/16 (224×224)</b></td>
    </tr>
    <tr>
      <td>Test Accuracy (10,000 images)</td>
      <td align="center">96.29%</td>
      <td align="center"><b>98.42%</b></td>
    </tr>
    <tr>
      <td>Model Size</td>
      <td align="center">45.0 MB (<code>.h5</code>)</td>
      <td align="center"><b>22.4 MB (<code>.onnx</code>)</b></td>
    </tr>
    <tr>
      <td>Runtime</td>
      <td align="center">TensorFlow + ONNX</td>
      <td align="center"><b>ONNX Runtime</b></td>
    </tr>
  </tbody>
</table>

## Confusion Matrix

![CIFAR 10 Confusion Matrix](results/confusion_matrix_tta.png)

* Metrics: [`results/metrics.json`](results/metrics.json)
* Checksum: [`results/model_sha256.txt`](results/model_sha256.txt)
* Verify: `python scripts/eval_cifar_test.py`

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests with `pip install -r requirements-dev.txt && pytest`.

## License

Released under the [MIT License](LICENSE).
