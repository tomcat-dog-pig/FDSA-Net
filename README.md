

# Dual‑Domain Adaptive Fusion with Dynamic Sparse Attention for Low‑Light Image Enhancement
This repository contains the official PyTorch implementation for the manuscript submitted to The Visual Computer.
## Overview
<img src="figures/network.png" width="700">

## Environment Setup
*   Python 3.8+
*   PyTorch 1.10.0+ 
*   CUDA 11.3+
*   Other dependencies: `pip install -r requirements.txt`

## Dataset Preparation
We evaluate our FDSA-Net on several widely used low-light image enhancement datasets. You can download the datasets from their official websites:

* **LOL Dataset (LOL-v1 & LOL-v2):** [Official BMVC 2018 Website](https://daooshee.github.io/BMVC2018website)
  
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
```
## Train
make sure your datasets are correctly placed and run:
```bash
python train.py
```
## Test
```bash
python test.py --weights ./checkpoints/best_model.pth --test_dir ./dataset/LOLv1/Test/input/
```
## Inference
test result on LOLv1&LOLv2

<img src="figures/result_LOLv1.png" width="500">
<img src="figures/result_LOLv2.png" width="500">


