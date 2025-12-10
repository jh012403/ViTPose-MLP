# Keypoint Processing Pipeline

This module implements a comprehensive preprocessing pipeline that leverages **ViTPose** to extract 2D keypoints (COCO-17) from RGB imagery. Subsequently, the data undergoes topology adaptation to the MPI-compatible **COCO-15** format and a series of normalization steps to generate optimized inputs for the training phase.

---

## Directory Structure

The pipeline is modularized into three distinct stages:

- `step1_extract_and_convert.py`: Performs backbone inference using ViTPose and executes the topology adaptation (COCO17 $\rightarrow$ COCO15).
- `step2_center_align_and_scale.py`: Handles coordinate canonicalization, including root-centering and scale normalization based on shoulder distance.
- `step3_normalize_final_format.py`: Applies final global normalization to the range $[-1, 1]$ and serializes the training artifacts.

---

## Processing Workflow

1.  **ViTPose Inference**: Extraction of 2D keypoints using the ViTPose backbone (COCO-17 topology).
2.  **Topology Adaptation (COCO-15)**: Restructuring of the skeleton by interpolating the `neck`, `spine`, and `head_top` joints to align with the MPI-INF-3DHP definition.
3.  **Resolution Normalization**: Coordinates are normalized relative to the input resolution ($192 \times 256$).
4.  **Canonicalization**:
    * **Centering**: Alignment of the skeleton based on the `spine` joint.
    * **Scaling**: Rescaling based on the Euclidean distance between shoulders.
5.  **Global Normalization**: Final scaling of coordinates to the range $[-1, 1]$ using the absolute maximum value, followed by data serialization.

---

## Detailed Script Descriptions

### 1️⃣ step1_extract_and_convert.py

**Functionality**:
- **Inference**: Loads the ViTPose configuration and checkpoint to perform 2D pose estimation on raw images.
- **Adaptation**: Converts the standard 17-keypoint output to the 15-keypoint format required by the project.
- **Preprocessing**: Applies initial normalization based on image resolution and saves the intermediate results.

**I/O Specification**:
- **Input**: Raw images located in `npzimage/*.png`.
- **Output**: Serialized NumPy archives at `npzdata/[DatasetName]/[DatasetName]_keypoints15.npz`.

**Usage**:
```bash
python step1_extract_and_convert.py
```
