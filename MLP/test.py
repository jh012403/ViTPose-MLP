import os
import numpy as np
import torch
from architecture.MLP import MLP

# 경로 설정
input_root = "npzdata"
output_root = "infernpz"
os.makedirs(output_root, exist_ok=True)

# MLP 모델 로드
model = MLP()
model.load_state_dict(torch.load(
    "ViTPose-Cus/MLP/checkpoints/7st/7st.pth",
    map_location="cpu"
))
model.eval()

# 100개 파일 순회
subdirs = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])

for subdir in subdirs:
    try:
        input_path = os.path.join(input_root, subdir, f"{subdir}.npz")
        output_path = os.path.join(output_root, f"{subdir}_infer.npz")

        if not os.path.exists(input_path):
            print(f"❌ 입력 누락: {input_path}")
            continue

        # 1. Load 2D keypoints
        data = np.load(input_path)
        input_2d = data["keypoints"].reshape(1, -1)  # (1, 30)

        # 🔽 추가: 정규화 정보 불러오기
        center = data["center"]
        shoulder_len = data["shoulder_len"]
        max_abs = data["max_abs"]

        # 2. MLP 추론
        input_tensor = torch.tensor(input_2d, dtype=torch.float32)
        with torch.no_grad():
            output_3d = model(input_tensor).numpy().reshape(15, 3)

        # 3. 후처리
        output_3d[:, [0, 2]] = output_3d[:, [2, 0]]     # X ↔ Z
        output_3d[:, 2] *= -1                           # Z 축 뒤집기
        foot_y_avg = (output_3d[11, 1] + output_3d[12, 1]) / 2
        output_3d[:, 1] -= foot_y_avg                   # 발 기준 Y 정렬

        # 4. 저장 (.npz) - 정규화 정보 포함
        np.savez(output_path,
                 keypoints_3d=output_3d.astype(np.float32),
                 center=center.astype(np.float32),
                 shoulder_len=shoulder_len.astype(np.float32),
                 max_abs=max_abs.astype(np.float32))
        
        print(f"✅ 저장 완료: {output_path}")

    except Exception as e:
        print(f"❌ 오류 발생: {subdir} → {e}")