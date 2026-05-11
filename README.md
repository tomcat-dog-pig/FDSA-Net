

# Dual‑Domain Adaptive Fusion with Dynamic Sparse Attention for Low‑Light Image Enhancement
This repository contains the official PyTorch implementation for the manuscript submitted to The Visual Computer.

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20123628-blue.svg)](https://doi.org/10.5281/zenodo.20123628)

## Overview
<img src="figures/network.png" width="700">

## Environment Setup
*   Python 3.8+
*   PyTorch 1.10.0+ 
*   CUDA 11.3+
*   Other dependencies: `pip install -r requirements.txt`

## Dataset Preparation
We evaluate our FDSA-Net on several widely used low-light image enhancement datasets. You can download the datasets from their official websites and put them in data:

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
python test.py 
```
## Inference
test result on LOLv1&LOLv2

<img src="figures/result_LOLv1.png" width="500">
<img src="figures/result_LOLv2.png" width="500">

## Citation

If you find our work is useful for your research, please consider citing our manuscript submitted to **The Visual Computer**:

```bibtex
@article{zhou2026fdsanet,
  title={Dual-Domain Adaptive Fusion with Dynamic Sparse Attention for Low-Light Image Enhancement},
  author={Zhou, Huaping and Chen, Hong and Sun, Kelei},
  journal={The Visual Computer},
  year={2026},
  note={Under Review}
}

