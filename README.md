-----

# 🧠 2D-to-3D Human Pose Estimation Pipeline

This project is a **lightweight pose estimation pipeline** that takes a single RGB image as input, extracts 2D keypoints, and infers a 3D pose based on them.
The entire pipeline consists of a ViTPose-based 2D pose estimator and an MLP-based 3D regressor incorporating SE-Residual Blocks.

-----

## 🔁 Pipeline Overview

```text
RGB Image
   ↓
[ViTPose]
   ↓             (COCO17 keypoints)
[COCO15 Conversion & Normalization]
   ↓             (COCO15 keypoints)
[MLP Regressor]
   ↓
3D Pose Estimation Result (15 × 3)
```

-----

## 📁 Directory Structure

```
project-root/
├── ViTPose/                 # ViTPose-based 2D keypoint inference and conversion
│   ├── Model/
│   ├── COCO15_convert/
│   ├── pretrained/checkpoints/
│   └── README.md
│
├── MLP/                     # SE-Residual based MLP 3D regressor
│   ├── model.py
│   └── README.md
│
├── dataset/                 # Dataset info for training/inference and preprocessing results
│   └── README.md
│
├── requirements.txt         # (Optional) Full environment configuration file
└── README.md                # (This document)
```

-----

## 🧩 Component Description

### 🔹 ViTPose

  - [MMPose](https://github.com/open-mmlab/mmpose) based 2D pose estimator.
  - Infers keypoints in COCO17 format.
  - Post-processing includes conversion to MPI-style COCO15 format and normalization.

### 🔹 COCO15 Converter

  - Constructs COCO15 format by interpolating points such as the neck, spine, and head\_top based on ViTPose output (COCO17).
  - Performs normalization based on image resolution.

### 🔹 MLP (3D Pose Regressor)

  - **Input:** 2D keypoints of 15 joints (Normalized 30-dimensional vector).
  - **Output:** 3D keypoints (45-dimensional).
  - **Structure:** MLP architecture incorporating SEBlock + Residual connections.
  - Designed to balance lightweight structure with accuracy.

-----

## 📦 Dataset Configuration

  - **Input for 2D Inference:** RGB images from MPI-INF-3DHP.
  - **Train/Test Split:**
      - Train: S1 \~ S4
      - Test: S5 \~ S7
  - **2D Keypoint Inference:** ViTPose + COCO15 Converter.
  - **MLP Training Input:** Normalized `.npz` files (15×2 keypoints → Center alignment, Shoulder scaling, [-1, 1] normalization).

-----

## 📝 Notes

  - The goal of this project is to implement 3D pose estimation based on a single image using a lightweight structure.
  - ViTPose utilizes pretrained models (`.pth`), and inference is performed based on the MMPose API.
  - The MLP structure combines **SEBlock**, **Residual**, and **GELU** to improve performance compared to standard Vanilla MLPs.
