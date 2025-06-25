# 🧠 Keypoint Converter Pipeline

이 프로젝트는 ViTPose를 활용하여 RGB 이미지에서 2D COCO17 keypoint를 추출한 후, MPI 기반 COCO15 포맷으로 변환하고 학습에 적합하도록 정규화하는 전처리 파이프라인입니다.

---

## 📁 디렉토리 구성

- `step1_extract_and_convert.py`: ViTPose 기반 keypoint 추출 및 COCO15 변환
- `step2_center_align_and_scale.py`: 중심 정렬 및 어깨 기준 정규화
- `step3_normalize_final_format.py`: [-1, 1] 범위 정규화 및 학습용 최종 저장

---

## 🔄 처리 흐름 요약

1. **ViTPose 추론**: 입력 이미지를 통해 2D keypoint(COCO17) 추출
2. **COCO15 변환**: neck, spine, head_top 보간 → 15개 keypoint 구성
3. **정규화 1**: 입력 해상도 기준 (192x256) 정규화
4. **정규화 2**: spine 기준 중심 정렬 + 어깨 거리 스케일링
5. **정규화 3**: 절댓값 최대값 기준 [-1, 1] 정규화 및 추가 정보 저장

---

## 🧩 단계별 스크립트 설명

### 1️⃣ step1_extract_and_convert.py

**기능**:
- ViTPose 추론 (config & checkpoint 필요)
- 17개 keypoint → 15개로 변환
- 해상도 기준 정규화 및 저장

**입출력**:
- 입력: `npzimage/*.png`
- 출력: `npzdata/[이름]/[이름]_keypoints15.npz`

**실행 예**:
```bash
python step1_extract_and_convert.py
```