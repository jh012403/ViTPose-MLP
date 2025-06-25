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
# IMG_WIDTH, IMG_HEIGHT = 192, 256

# === COCO-17 → MPI 기반 COCO-15 변환 함수 ===
def convert_coco17_to_coco15(keypoints_17):
    # 필요한 keypoint 추출
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

    # === Neck: shoulder midpoint → nose 방향으로 30% 보간
    mid_shoulder = (l_shoulder + r_shoulder) / 2
    neck = mid_shoulder + 0.3 * (nose - mid_shoulder)

    # === Spine: mid_hip → mid_shoulder 사이 40% 상단 보간
    mid_hip = (l_hip + r_hip) / 2
    spine_vec = mid_shoulder - mid_hip
    spine = mid_hip + 0.4 * spine_vec

    # === HeadTop: nose 대체
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

# === 경로 설정 ===
image_path = 'ViTPose-Cus/test_image/2d_keypoint/sample_image/side_leternel_raise.png'
save_dir = 'ViTPose-Cus/MLP/mlp_input/'
os.makedirs(save_dir, exist_ok=True)

# === 이미지 로드 및 추론 ===
image = cv2.imread(image_path)
if image is None:
    raise RuntimeError(f"❌ 이미지 로드 실패: {image_path}")

result = inference_topdown(model, image_path)
if len(result) == 0 or result[0].pred_instances.keypoints.shape[0] == 0:
    raise RuntimeError(f"⚠️ Keypoint 없음: {image_path}")

# === Keypoint 변환 ===
keypoints_17 = result[0].pred_instances.keypoints[0]  # (17, 2)
keypoints_15 = convert_coco17_to_coco15(keypoints_17)  # (15, 2)

# === 정규화 (min-max → -1~1 스케일링)
x_min, x_max = 110.02, 597.28
y_min, y_max = 113.06, 686.26

keypoints_15[:, 0] = (keypoints_15[:, 0] - x_min) / (x_max - x_min)
keypoints_15[:, 1] = (keypoints_15[:, 1] - y_min) / (y_max - y_min)
keypoints_15 = (keypoints_15 - 0.5) * 2

# === 저장 ===
save_name = os.path.splitext(os.path.basename(image_path))[0]
save_path = os.path.join(save_dir, f"{save_name}_keypoints15_test.npz")
np.savez(save_path, keypoints=keypoints_15, source_flag=np.array([-1], dtype=np.int32))

print(f"✅ 저장 완료: {save_path}")