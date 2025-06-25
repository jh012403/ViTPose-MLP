import torch
import time
from architecture.MLP import MLP  # 너의 모델 경로에 맞게 조정

device = torch.device("cuda:0")
model = MLP().to(device)
model.load_state_dict(torch.load("ViTPose-Cus/MLP/checkpoints/7st/7st.pth", map_location=device))
model.eval()

input_tensor = torch.randn(1, 30).to(device)

# Warmup
for _ in range(50):
    _ = model(input_tensor)

# 실측
repeat = 500
start = time.time()
with torch.no_grad():
    for _ in range(repeat):
        _ = model(input_tensor)
end = time.time()

fps = repeat / (end - start)
print(f"✅ A4000 기준 FPS: {fps:.2f}")