import os
import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown

# === 모델 설정 ===
config_file = 'ViTPose-Cus/configs/td-hm_ViTPose-small_120e.py'
checkpoint_file = 'ViTPose-Cus/checkpoints/ViTPose-s_120e.pth'
model = init_model(config_file, checkpoint_file, device='cuda:1')

# === pipeline 수동 설정 ===
model.cfg.test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBboxCenterScale'),
    dict(type='ResizeBBox', scale_factor=1.25),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]

# === ViTPose 이미지 정규화 기준 해상도 ===
IMG_WIDTH, IMG_HEIGHT = 192, 256

# === COCO-17 → MPI 기반 COCO-15 변환 함수 ===
def convert_coco17_to_coco15(keypoints_17):
    nose        = keypoints_17[0]
    l_shoulder  = keypoints_17[5]
    r_shoulder  = keypoints_17[6]
    l_elbow     = keypoints_17[7]
    r_elbow     = keypoints_17[8]
    l_wrist     = keypoints_17[9]
    r_wrist     = keypoints_17[10]
    l_hip       = keypoints_17[11]
    r_hip       = keypoints_17[12]
    l_knee      = keypoints_17[13]
    r_knee      = keypoints_17[14]
    l_ankle     = keypoints_17[15]
    r_ankle     = keypoints_17[16]

    mid_shoulder = (l_shoulder + r_shoulder) / 2
    neck = mid_shoulder + 0.3 * (nose - mid_shoulder)

    mid_hip = (l_hip + r_hip) / 2
    spine_vec = mid_shoulder - mid_hip
    spine = mid_hip + 0.4 * spine_vec

    head_top = nose

    keypoints_15 = np.vstack([
        neck,        # 0
        l_shoulder,  # 1
        r_shoulder,  # 2
        l_elbow,     # 3
        r_elbow,     # 4
        l_wrist,     # 5
        r_wrist,     # 6
        l_hip,       # 7
        r_hip,       # 8
        l_knee,      # 9
        r_knee,      # 10
        l_ankle,     # 11
        r_ankle,     # 12
        spine,       # 13
        head_top     # 14
    ])
    return keypoints_15

# === 디렉토리 설정 ===
image_dir = 'npzimage'
save_root = 'npzdata'
os.makedirs(save_root, exist_ok=True)

# === 이미지 전체 순회 ===
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ 이미지 로드 실패: {image_path}")
        continue

    try:
        result = inference_topdown(model, image_path)
        if len(result) == 0 or result[0].pred_instances.keypoints.shape[0] == 0:
            print(f"⚠️ Keypoint 없음: {image_path}")
            continue

        keypoints_17 = result[0].pred_instances.keypoints[0]
        keypoints_15 = convert_coco17_to_coco15(keypoints_17)

        # === 정규화 ===
        keypoints_15[:, 0] /= IMG_WIDTH
        keypoints_15[:, 1] /= IMG_HEIGHT

        # === 저장 디렉토리 생성 ===
        save_dir = os.path.join(save_root, os.path.splitext(image_file)[0])
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{os.path.splitext(image_file)[0]}_keypoints15.npz")
        np.savez(save_path, keypoints=keypoints_15, source_flag=np.array([-1], dtype=np.int32))
        print(f"✅ 저장 완료: {save_path}")

    except Exception as e:
        print(f"❌ 오류 발생: {image_path} → {e}")