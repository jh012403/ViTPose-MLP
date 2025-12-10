## Dataset Organization

This project utilizes a curated dataset pipeline optimized for 2D and 3D pose estimation tasks. The data preparation process involves utilizing standard benchmarks for backbone training and specific regression tasks, with a focus on format consistency across different domains.

### Source Datasets

#### MPI-INF-3DHP
- **Role**: Primary dataset for training and evaluating the **MLP-based 3D Regression Network**.
- **Protocol & Splits**:
  - **Training Set**: Subjects `S1`, `S2`, `S3`, `S4`
  - **Test Set**: Subjects `S5`, `S6`, `S7`
- **Preprocessing Pipeline**:
  1. **Frame Extraction**: RGB frames are extracted from the source video sequences.
  2. **2D Inference**: Keypoint inference is performed using **ViTPose**.
  3. **Serialization**: Inferred keypoints are mapped to the **COCO-15** format and serialized into `.npz` files for efficient loading.

#### COCO17
- **Role**: Source dataset for training the 2D pose estimator (**ViTPose**).
- **Specification**: Follows the standard 17-keypoint COCO topology.
- **Keypoint Adaptation**:
  - To ensure compatibility with the MPI-INF-3DHP topology, we employ a **Keypoint Adapter (COCO17 → COCO15)**.
  - This process reduces the standard 17 joints to a 15-joint subset, aligning the input dimension for the subsequent 3D lifting stage.

---

### Data Generation Pipeline

The overall data flow—from the 2D backbone pre-training to the final 3D regression input preparation—is summarized below:

```text
[ Data Processing Workflow ]

1. Backbone Training
   Dataset: COCO17 (17 Keypoints)
      ↓
   Model: ViTPose (Pre-trained Weights)

2. Inference & Adaptation
   Input: MPI-INF-3DHP (RGB Images)
      ↓
   Inference: ViTPose estimates 2D poses
      ↓
   Adaptation: Keypoint Mapping (COCO17 → COCO15)
      ↓
   Preprocessing: Normalization & Serialization (.npz)
      ↓
   Target: Input features for MLP 3D Regression Training
```
