

# FDSA-Net: A frequency-guided dynamic sparse attention network with dual-domain fusion for low-light image enhancement

## Overview
![Architecture](figures/network.png) 


## Environment Setup
*   Python 3.8+
*   PyTorch 1.10.0+ 
*   CUDA 11.3+
*   Other dependencies: `pip install -r requirements.txt`

## Dataset Preparation
Please download the standard low-light enhancement datasets:
*   **LOL-v1**: [Download Link]
*   **LOL-v2**: [Download Link]

Organize the datasets in the following structure for `dataloader.py`:
```text
data/
├── LOLv1/
│   ├── Train/
│   │   ├── input/
│   │   └── target/
│   └── Test/
│       ├── input/
│       └── target/
├── LOLv2/
│   ├── Train/
│   │   ├── input/
│   │   └── target/
│   └── Test/
│       ├── input/
│       └── target/
