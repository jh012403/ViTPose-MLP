import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Reproducibility용 시드 설정 함수
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed set to {seed}")

def save_checkpoint(model, path: str):
    """
    모델 저장 함수 (.pt)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")

def load_checkpoint(model, path: str):
    """
    저장된 모델을 불러오는 함수
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ Checkpoint file not found at: {path}")
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    print(f"📦 Loaded checkpoint from {path}")
    return model