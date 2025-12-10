# 🧠 MLP-based 3D Pose Regressor

This module is a lightweight MLP-based regression model that takes 2D COCO-15 keypoints as input and outputs 3D human pose coordinates. Its core component is the **SE-Residual Block**, which combines Squeeze-and-Excitation with residual connections.

---

## ✅ Key Feature: SE-Residual Block

- **Squeeze-and-Excitation (SE)** recalibrates joint-wise feature importance dynamically.  
- **Residual connections** minimize information loss and improve training stability.  
- Uses **GELU activation** for smooth and expressive nonlinear transformation.

---

## 🧩 Network Architecture Overview

- **Input:** 2D keypoints (15 joints × 2D = 30 dimensions)  
- **Output:** 3D keypoints (15 joints × 3D = 45 dimensions)

```text
Input (30)
  ↓
Input Layer (Linear → BN → GELU → Dropout)
  ↓
ResidualSEBlock (1024)
  ↓
Linear → GELU → ResidualSEBlock (512)
  ↓
Linear → GELU → ResidualSEBlock (256)
  ↓
Bottleneck (Linear → GELU → Dropout)
  ↓
Output Layer (Linear → 45)
```

---

## 📂 Main Files

- **`model.py`** – Definition of the MLP network  
  - `SEBlock`: Nonlinear attention block for joint-level importance weighting  
  - `ResidualSEBlock`: Combination of SE and residual pathways  
  - `MLP`: Full regression architecture

---

## 🧪 Input Data Format

- **Location:** `npzdata/[name].npz`  
- **Example structure:**

```python
import numpy as np
data = np.load('npzdata/sample_001.npz')
x = data['keypoints'].reshape(1, -1)  # shape: (1, 30)
```

- Included fields:
  - `keypoints`: normalized (15, 2) 2D coordinates  
  - `center`, `shoulder_len`, `max_abs`: metadata for post-processing reconstruction

---

## ⚙️ Inference Example

```python
from model import MLP
import torch

model = MLP()
x = torch.randn(1, 30)      # (batch size, 15 keypoints × 2)
y = model(x)                # output shape: (1, 45)
```

---

## 📝 Notes

- Dropout rate is set to **0.35** by default and applied to the input and each block.  
- The output 3D keypoints are normalized and must be denormalized using metadata to recover actual scale and position.  
- The architecture is designed to balance **lightweight efficiency** and **accuracy**.
