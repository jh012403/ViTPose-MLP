# config.yaml
seed: 42

data_path: /app_data/home/jonghyun/jhworkspace/ViTPose-Cus/data/custom/1merged_3dpw_mpi_normed.npz
save_path: /app_data/home/jonghyun/jhworkspace/ViTPose-Cus/MLP/checkpoints/5st/5st.pth
log_dir: /app_data/home/jonghyun/jhworkspace/ViTPose-Cus/MLP/runs/mlp_log/5st/

train_split: 0.8
batch_size: 64          # per GPU (Total = 64 x 3 = 192)
epochs: 300
lr: 0.0005              # = 5e-4

scheduler:
  T_0: 300        # 전체 에폭 주기를 1회로 고정
  T_mult: 1       # 멀티플라이 X, 단일 사이클
  eta_max: 0.0005
  eta_min: 1e-6   # (config에 빠져 있으면 default 추가 필요)
  T_up: 30        # 전체 에폭의 10% 정도 warm-up 권장
  gamma: 0.5

  # model architecture
  import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.GELU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x).unsqueeze(1)
        return x * scale.squeeze(1)

class ResidualSEBlock(nn.Module):
    def __init__(self, dim, dropout=0.35):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.se = SEBlock(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.se(out)
        return F.gelu(out + residual)

class MLP(nn.Module):
    def __init__(self, input_dim=28, output_dim=42, hidden_dims=[1024, 512, 256], dropout=0.35):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.res_blocks = nn.Sequential(
            ResidualSEBlock(hidden_dims[0], dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            ResidualSEBlock(hidden_dims[1], dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.GELU(),
            ResidualSEBlock(hidden_dims[2], dropout)
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2] // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.output_layer = nn.Linear(hidden_dims[2] // 2, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.bottleneck(x)
        return self.output_layer(x)
