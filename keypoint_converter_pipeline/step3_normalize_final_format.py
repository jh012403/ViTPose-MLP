import numpy as np
import os

# === 상위 디렉토리 설정 ===
root_dir = "npzdata"
subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

for subdir in subdirs:
    try:
        input_path = os.path.join(root_dir, subdir, f"{subdir}_ad.npz")
        output_path = os.path.join(root_dir, subdir, f"{subdir}.npz")

        if not os.path.exists(input_path):
            print(f"❌ 입력 파일 없음: {input_path}")
            continue

        # === 1. Load
        data = np.load(input_path)
        keypoints = data["keypoints"].astype(np.float32)  # shape: (15, 2)

        # === 2. 중심 정렬 (Spine 기준: index 13)
        center = keypoints[13]
        keypoints_centered = keypoints - center

        # === 3. 어깨 거리 기준 스케일 정규화
        shoulder_len = np.linalg.norm(keypoints_centered[1] - keypoints_centered[2])
        if shoulder_len == 0:
            print(f"⚠️ 어깨 너비 0: {input_path}")
            continue
        keypoints_scaled = keypoints_centered / shoulder_len

        # === 4. [-1, 1] 정규화 (절댓값 최대값 기준)
        max_abs = np.max(np.abs(keypoints_scaled))
        if max_abs == 0:
            print(f"⚠️ max_abs 0: {input_path}")
            continue
        keypoints_normalized = keypoints_scaled / max_abs

        # === 5. 저장 (정규화 정보 포함)
        np.savez(output_path,
                 keypoints=keypoints_normalized.astype(np.float32),
                 center=center.astype(np.float32),
                 shoulder_len=np.array([shoulder_len], dtype=np.float32),
                 max_abs=np.array([max_abs], dtype=np.float32))
        
        print(f"✅ 저장 완료 (정규화 정보 포함): {output_path}")

    except Exception as e:
        print(f"❌ 오류 발생: {subdir} → {e}")