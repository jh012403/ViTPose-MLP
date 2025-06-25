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
    def __init__(self, input_dim=30, output_dim=45, hidden_dims=[1024, 512, 256], dropout=0.35):
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
