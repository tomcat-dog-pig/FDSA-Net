

# FDSA-Net: A frequency-guided dynamic sparse attention network with dual-domain fusion for low-light image enhancement
The official PyTorch implementation of the paper "**FDSA-Net: A frequency-guided dynamic sparse attention network with dual-domain fusion for low-light image enhancement**" .

## Abstract
Existing Transformer-based low-light image enhancement methods typically compute attention in the channel dimension. While this reduces computational complexity, it results in spatial information loss, leading to color distortion in enhanced images. Additionally, most approaches neglect the role of frequency information in preserving image details. To address these issues, we propose a frequency-guided dynamic sparse attention network (FDSA-Net) for low-light image enhancement. First, we design an Axis-based Dynamic Sparse Attention (ADSA) block to efficiently capture long-range dependencies without sacrificing spatial context. Specifically, we compute attention with an axis-based one and incorporate a dynamic top-k mechanism to retain the most informative tokens. Second, we introduce a frequency-domain branch, which utilizes several Frequency Processing (FP) blocks to achieve superior detail restoration. Additionally, we construct a Multi-scale Dual-domain Adaptive Fusion (MDAF) module that effectively integrates spatial and frequency-domain features. Extensive experiments on several public datasets demonstrate that our proposed method surpasses other state-of-the-art methods across multiple quantitative metrics, achieving improvements of 0.014 in SSIM and 0.30 in PSNR on the LOL-v1 dataset.

## Overview
![Architecture](figures/network.png) 


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
