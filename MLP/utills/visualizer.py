# visualizer.py

from torch.utils.tensorboard import SummaryWriter
import os

class Visualizer:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, value, step):
        """한 스칼라 값 기록 (ex. train_loss, val_loss 등)"""
        self.writer.add_scalar(tag, value, step)

    def log_metrics(self, metrics: dict, step: int, prefix: str = ''):
        """
        여러 지표를 한 번에 기록
        metrics: {'MPJPE': 42.1, 'Loss': 1.23, ...}
        prefix: 'Train' or 'Val' 등
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def add_graph(self, model, input_tensor):
        """모델의 연산 그래프를 기록"""
        self.writer.add_graph(model, input_tensor)
    
    def close(self):
        self.writer.close()