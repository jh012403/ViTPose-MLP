# 🧷 ViTPose Inference Module

This module implements a preprocessing pipeline utilizing the [MMPose](https://github.com/open-mmlab/mmpose) framework. It deploys the **ViTPose** architecture to estimate 2D keypoints (COCO-17 topology) from RGB imagery and subsequently adapts them to the COCO-15 format required for downstream tasks.

---

## 📌 Executive Summary

- **Framework**: OpenMMLab MMPose
- **Model Architecture**: ViTPose-small (`ViTPose-s_120e.pth`, `vitpose_small.pth`)
- **Input Specification**: Single-view RGB image
- **Output Specification**: 2D Keypoints in COCO-17 or COCO-15 format (`numpy array` or `.npz`)
- **Primary Objective**: Generating input features for the MLP-based 3D Pose Estimation network.

---

## 📂 Directory Structure

```text
ViTPose/
├── COCO15_convert/
│   └── coco17_to_coco15.py         # Topology adaptation (17→15) & Normalization
│
├── mmpose/                         # MMPose library source (submodule or symlink)
│
├── Model/
│   └── td-hm_ViTPose-small_120e.py # Main inference script
│
├── pretrained/
│   └── checkpoints/                # Pre-trained model weights
│       ├── vitpose_small.pth
│       └── ViTPose-s_120e.pth
│
└── README.md                       # (Documentation)
```
