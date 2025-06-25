import cv2
import os
import random
from glob import glob

# 랜덤 시드 고정 (재현성 원할 경우)
random.seed(42)

# 대상 video_*.avi 파일들
video_paths = sorted(glob("ViTPose-Cus/data/mpi_inf_3dhp/S[6-8]/Seq1/imageSequence/video_*.avi"))

# 저장 디렉토리
save_path = "npzimage"
os.makedirs(save_path, exist_ok=True)

# 모든 avi 영상에서 랜덤하게 프레임 추출
saved_count = 0
target_count = 100

while saved_count < target_count:
    # 무작위로 하나의 영상 선택
    video_path = random.choice(video_paths)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 무작위 프레임 index 선택
    frame_index = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    cap.release()

    if ret:
        save_name = os.path.join(save_path, f"{saved_count + 1}st.png")
        cv2.imwrite(save_name, frame)
        print(f"✅ 저장 완료: {save_name} (video: {os.path.basename(video_path)}, frame: {frame_index})")
        saved_count += 1
    else:
        print(f"❌ 실패: {video_path}의 frame {frame_index}")