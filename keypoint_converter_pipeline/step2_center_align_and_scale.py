import os
import numpy as np

root_dir = "npzdata"
subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

for subdir in subdirs:
    try:
        npz_path = os.path.join(root_dir, subdir, f"{subdir}_keypoints15.npz")
        if not os.path.exists(npz_path):
            print(f"❌ 누락된 파일: {npz_path}")
            continue

        data = np.load(npz_path)
        keypoints = data["keypoints"]  # shape: (15, 2)

        # === 중심 이동 (spine 기준: index 13)
        center = keypoints[13]
        keypoints_centered = keypoints - center

        # === 어깨 너비로 정규화
        shoulder_width = np.linalg.norm(keypoints_centered[1] - keypoints_centered[2])
        if shoulder_width == 0:
            print(f"⚠️ 어깨 너비 0: {npz_path}")
            continue

        keypoints_mpi_style = keypoints_centered / shoulder_width

        # === 저장 경로
        save_path = os.path.join(root_dir, subdir, f"{subdir}_ad.npz")
        np.savez(save_path, 
                 keypoints=keypoints_mpi_style.astype(np.float32),
                 source_flag=np.array([-2], dtype=np.int32))

        print(f"✅ 저장 완료: {save_path}")

    except Exception as e:
        print(f"❌ 오류 발생: {subdir} → {e}")