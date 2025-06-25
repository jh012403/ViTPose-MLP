import torch
from thop import profile
import torch.nn as nn

# === MLP 모델 정의 ===
class MLP(nn.Module):
    def __init__(self, input_dim=30, output_dim=45, hidden_dims=[1024, 512, 256], dropout=0.25):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# === 모델 및 입력 설정 ===
model = MLP()
dummy_input = torch.randn(1, 30)

# === FLOPs / Params 계산 ===
flops, params = profile(model, inputs=(dummy_input,), verbose=False)
print(f"✅ FLOPs: {flops / 1e6:.2f} M")
print(f"✅ Params: {params / 1e6:.2f} M")