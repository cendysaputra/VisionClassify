# VisionClassify

Lightweight Streamlit and ONNX Runtime web app with **98.47%** CIFAR 10 test accuracy and **83.99%** Top 5 accuracy across **256** general object labels. Both numbers are measured one image at a time, as the app runs.

![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B)
![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX%20INT8-005CED)
![CIFAR 10](https://img.shields.io/badge/CIFAR%2010-98.47%25-2EA043)
![General Top 5](https://img.shields.io/badge/General%20Top%205-83.99%25-0969DA)
![License](https://img.shields.io/badge/License-MIT-blue)

## Overview

Upload any JPG, PNG, or WEBP photo to classify objects directly in your browser. Both inference modes run on a unified **ONNX Runtime** backend without TensorFlow for fast startup and low memory usage.

* **CIFAR 10 Mode**: Predicts 10 classic object categories using a quantized **ViT Small/16** model bundled in the repository.
* **General Mode**: Recognizes 256 everyday objects, places, animals, and scenes using zero shot **CLIP ViT B/16** with automatic low confidence detection.

## Screenshots

<table>
  <tr>
    <td align="left" width="50%">
      <img src="results/screenshot_cifar10.png" alt="CIFAR 10 Mode" width="100%" /><br />
      <sub><b>CIFAR 10 Mode</b>: ViT Small INT8 ONNX (10 classes)</sub>
    </td>
    <td align="left" width="50%">
      <img src="results/screenshot_umum.png" alt="General Mode" width="100%" /><br />
      <sub><b>General Mode</b>: CLIP ViT B/16 ONNX (256 labels)</sub>
    </td>
  </tr>
</table>

## CIFAR 10 Mode Performance

<table>
  <thead>
    <tr>
      <th align="left">Metric</th>
      <th align="left">v6</th>
      <th align="left">v7 (Current)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Model</td>
      <td align="left">ResNet 18 (32×32), trained from scratch on CIFAR 10</td>
      <td align="left"><b>ViT Small/16 (224×224), pretrained, then fine tuned on CIFAR 10</b></td>
    </tr>
    <tr>
      <td align="left">Test Accuracy (10,000 images)</td>
      <td align="left">96.29%</td>
      <td align="left"><b>98.47%</b> (9,847)</td>
    </tr>
    <tr>
      <td align="left">Model Size</td>
      <td align="left">45.0 MB (<code>.h5</code>)</td>
      <td align="left"><b>22.4 MB (INT8 <code>.onnx</code>)</b></td>
    </tr>
    <tr>
      <td align="left">Runtime</td>
      <td align="left">TensorFlow + ONNX</td>
      <td align="left"><b>ONNX Runtime</b></td>
    </tr>
  </tbody>
</table>

The two models are not trained the same way: v7 starts from weights pretrained on a large image dataset, v6 saw only CIFAR 10. Keep that in mind when comparing them.

![CIFAR 10 Confusion Matrix](results/confusion_matrix_tta.png)

**Source.** v7 is [`MF21377197/vit-small-patch16-224-finetuned-Cifar10`](https://huggingface.co/MF21377197/vit-small-patch16-224-finetuned-Cifar10) (Apache 2.0), a 3 epoch fine tune of [`WinKawaks/vit-small-patch16-224`](https://huggingface.co/WinKawaks/vit-small-patch16-224), exported to ONNX and quantized to INT8. Its training log ([`results/provenance/`](results/provenance/)) shows 351 steps of 128 images per epoch and a 5,000 image evaluation set, so it trained on 45,000 training images and chose its checkpoint on the other 5,000. The 10,000 test images were not part of either.

**Batch effect.** The INT8 model quantizes activations per batch, so a prediction can change with the other images in the same batch. The app classifies one image per call, and the accuracy above is measured the same way. Batched runs gave 9,836 to 9,843. INT8 kernels also differ between CPU types, so another machine can differ by a few images.

* Metrics: [`results/metrics.json`](results/metrics.json)
* Checksum: [`results/model_sha256.txt`](results/model_sha256.txt)
* Verify: `python scripts/eval_cifar_test.py`, or the **Verify model accuracy** workflow, which also checks with code independent of the app

## General Mode Performance

Zero shot CLIP ViT B/16 (INT8 ONNX) on 1,530 CIFAR 100 test images (30 per class) from the 51 classes that also appear in the 256 labels. CIFAR 100 images are 32×32 and upscaled, so real photos behave differently.

<table>
  <thead>
    <tr>
      <th align="left">Metric</th>
      <th align="left">CLIP ViT B/32 (earlier)</th>
      <th align="left">CLIP ViT B/16 (Current)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">256 Label Top 1</td>
      <td align="left">55.03%</td>
      <td align="left"><b>61.70%</b></td>
    </tr>
    <tr>
      <td align="left">51 Class Top 1</td>
      <td align="left">70.92%</td>
      <td align="left"><b>73.92%</b></td>
    </tr>
    <tr>
      <td align="left">256 Label Top 5</td>
      <td align="left">81.70%</td>
      <td align="left"><b>83.99%</b></td>
    </tr>
    <tr>
      <td align="left">Vision Weights Size</td>
      <td align="left">89.1 MB</td>
      <td align="left"><b>87.5 MB</b></td>
    </tr>
  </tbody>
</table>

The B/32 column is carried over from the commit that switched to B/16 and was not re measured. The B/16 column is measured one image at a time. The same pack scored differently on different GitHub runners (936 and 930 of 1,530 for 256 label Top 1 in two runs), so differences below about half a point are noise.

The label embeddings in `data/general_label_embeddings.npz` are built by `scripts/build_general_embeddings.py` from the prompts in `general_labels.py`, and the **Verify General mode** workflow commits whatever that script builds. In the same run the previous pack (built outside the repo) scored 930, 1,287 and 1,128 for the three rows, and a seven template ensemble scored 928, 1,286 and 1,121; see [`results/general_comparison.json`](results/general_comparison.json). These differences are within the noise above.

* Metrics: [`results/general_metrics.json`](results/general_metrics.json)
* Verify: `python scripts/eval_general.py`, or the **Verify General mode** workflow

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run tests with `pip install -r requirements-dev.txt && pytest`.

## Contributing

AI coding agents and people alike: read [AGENTS.md](AGENTS.md) first. Changes go through pull requests, and accuracy numbers come from the verify workflows.

## License

Released under the [MIT License](LICENSE). The CIFAR 10 model weights are Apache 2.0 (see Source above); the CLIP weights are downloaded at runtime from [`Xenova/clip-vit-base-patch16`](https://huggingface.co/Xenova/clip-vit-base-patch16).
