# config.yaml
seed: 42

data_path: /app_data/home/jonghyun/jhworkspace/ViTPose-Cus/data/custom/merged_3dpw_mpi.npz
save_path: /app_data/home/jonghyun/jhworkspace/ViTPose-Cus/MLP/checkpoints/3st/3st.pth
log_dir: /app_data/home/jonghyun/jhworkspace/ViTPose-Cus/MLP/runs/mlp_log

train_split: 0.8
batch_size: 64          
epochs: 300
lr: 0.0005              

scheduler:
  T_0: 150              # 150 epoch마다 cosine 주기
  T_mult: 1             # 주기 고정
  eta_max: 0.0005       # 초기 lr
  eta_min: 1e-6         # 최소 lr
  T_up: 10              # warmup 단계
  gamma: 0.5            # 이후 주기 학습률 감쇠 비율 (T_mult>1일 때 사용)

# model architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.25):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.relu(out + residual)

class MLP(nn.Module):
    def __init__(self, input_dim=28, output_dim=42, hidden_dims=[1024, 512, 256], dropout=0.25):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dims[0], dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            ResidualBlock(hidden_dims[1], dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            ResidualBlock(hidden_dims[2], dropout)
        )

        self.output_layer = nn.Linear(hidden_dims[2], output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)