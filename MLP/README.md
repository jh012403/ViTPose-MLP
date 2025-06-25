# 🧠 MLP-based 3D Pose Regressor

본 모듈은 2D COCO15 keypoint를 입력으로 받아 3D 자세를 회귀하는 MLP 기반의 경량 회귀 모델입니다. 핵심 특징은 SE(Feature Recalibration) 기법과 Residual 연결을 결합한 **SE-Residual Block 구조**입니다.

---

## ✅ 핵심 특징: SE-Residual Block

- **Squeeze-and-Excitation (SE)** 구조를 통해 joint 간 중요도를 동적으로 조절
- **Residual 연결**로 정보 손실을 최소화하며 학습 안정성 향상
- **GELU 활성화 함수** 사용으로 부드러운 비선형 표현력 확보

---

## 🧩 네트워크 구조 요약

- 입력: 2D keypoint (15개 관절 × 2D = 30 차원)
- 출력: 3D keypoint (15개 관절 × 3D = 45 차원)

```
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

## 📂 주요 파일

- `model.py`: MLP 네트워크 정의
  - `SEBlock`: 관절별 중요도 조절을 위한 비선형 attention block
  - `ResidualSEBlock`: SE와 잔차 연결을 결합한 핵심 블록
  - `MLP`: 전체 회귀기 구조 구현

---

## 🧪 입력 데이터 포맷

- 위치: `npzdata/[이름].npz`
- `.npz` 파일 구성 예시:

```python
import numpy as np
data = np.load('npzdata/sample_001.npz')
x = data['keypoints'].reshape(1, -1)  # shape: (1, 30)
```

- 포함 필드:
  - `keypoints`: 정규화된 (15, 2) 형태의 2D 좌표
  - `center`, `shoulder_len`, `max_abs`: 후처리 복원용 메타데이터

---

## ⚙️ 실행 예시 (Inference)

```python
from model import MLP
import torch

model = MLP()
x = torch.randn(1, 30)      # (배치 크기, 15 keypoints × 2)
y = model(x)                # 출력 shape: (1, 45)
```

---

## 📝 참고 사항

- Dropout 비율은 기본적으로 0.35로 설정되어 있으며, 입력 및 각 블록에 적용됩니다.
- 출력된 3D keypoints는 정규화된 상태이며, 후처리를 통해 실제 크기 및 위치로 복원이 가능합니다.
- 전체 구조는 경량성과 정확도의 균형을 고려하여 설계되었습니다.
