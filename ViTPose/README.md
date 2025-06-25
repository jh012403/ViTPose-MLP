# 🧷 ViTPose Inference Module

이 디렉토리는 [MMPose](https://github.com/open-mmlab/mmpose) 기반의 **ViTPose** 모델을 사용하여 RGB 이미지에서 2D COCO17 keypoint를 추론하고, 이를 COCO15 포맷으로 변환하는 전처리 파이프라인입니다.

---

## 📌 요약

- **기반 프레임워크**: OpenMMLab MMPose
- **사용 모델**: ViTPose-small (`ViTPose-s_120e.pth`, `vitpose_small.pth`)
- **입력**: 단일 RGB 이미지
- **출력**: COCO17 또는 COCO15 포맷의 2D keypoints (`numpy array` 또는 `.npz`)
- **활용 목적**: MLP 기반 3D Pose Estimation의 입력 데이터 생성

---

## 📂 디렉토리 구성

```
ViTPose/
├── COCO15_convert/
│   └── coco17_to_coco15.py         # 17→15 keypoint 변환 및 정규화
│
├── mmpose/                         # MMPose 라이브러리 (clone 또는 symlink)
│
├── Model/
│   └── td-hm_ViTPose-small_120e.py # ViTPose 추론 스크립트
│
├── pretrained/
│   └── checkpoints/                # 사전학습 weight 저장 위치
│       ├── vitpose_small.pth
│       └── ViTPose-s_120e.pth
│
└── README.md                       # (본 문서)
```

---

## 🔁 처리 흐름 요약

```text
단일 이미지
   ↓
ViTPose 추론 (COCO17 keypoints)
   ↓
COCO15 변환 (neck, spine, head_top 보간)
   ↓
정규화 및 저장 (.npz)
   ↓
MLP 3D 회귀 입력으로 사용
```

---

## 📍 참고

- ViTPose는 Swin Transformer 백본을 기반으로 한 고성능 단일 이미지 2D 포즈 추정기입니다.
- 추론된 COCO17 keypoints는 후처리를 통해 MPI-style COCO15 포맷으로 변환됩니다.
- `mmpose/` 디렉토리는 [공식 GitHub 레포](https://github.com/open-mmlab/mmpose)에서 clone 하거나, symlink 형태로 연결해 구성할 수 있습니다.
