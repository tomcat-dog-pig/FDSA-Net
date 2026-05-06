

# FDSA-Net: A frequency-guided dynamic sparse attention network with dual-domain fusion for low-light image enhancement
The official PyTorch implementation of the paper "**FDSA-Net: A frequency-guided dynamic sparse attention network with dual-domain fusion for low-light image enhancement**" .

## Abstract
Low‑light image enhancement remains a critical task in computer vision and visual computing, supporting applications such as nighttime surveillance and autonomous driving. Existing Transformer‑based methods often suffer from spatial information loss or inefficient attention computation, while frequency‑domain approaches lack effective fusion with spatial features. To address these issues, we propose a frequency‑guided dynamic sparse attention network named **FDSA‑Net** for high‑quality low‑light enhancement. The model uses an axis‑based dynamic sparse attention block to capture long‑range dependencies efficiently and a frequency branch to preserve fine details. A multi‑scale dual‑domain adaptive fusion module unifies spatial and frequency features for robust enhancement. Extensive experiments on LOL‑v1, LOL‑v2, and several real‑world datasets show that our method achieves 24.41 dB PSNR and 0.868 SSIM on LOL‑v1, outperforming state‑of‑the‑art approaches. This work provides an effective dual‑domain solution for low‑light enhancement and offers insights for efficient visual computing models.

## Overview
![Architecture](figures/network.pdf) 
*(Note: Please upload your FDSA-Net architecture diagram to the `figures` folder and name it `architecture.png`)*

## Environment Setup
*   Python 3.8+
*   PyTorch 1.10.0+ 
*   CUDA 11.3+
*   Other dependencies: `pip install -r requirements.txt` *(You can create a requirements file or list packages like numpy, opencv-python, etc.)*

## Dataset Preparation
Please download the standard low-light enhancement datasets:
*   **LOL-v1**: [Download Link]
*   **LOL-v2**: [Download Link]

Organize the datasets in the following structure for `dataloader.py`:
```text
dataset/
├── LOLv1/
│   ├── Train/
│   │   ├── low/
│   │   └── high/
│   └── Test/
│       ├── low/
│       └── high/
