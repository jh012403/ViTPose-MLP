import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestartsWithPlateau(_LRScheduler):
    def __init__(self,
                 optimizer,
                 T_0,
                 T_mult=1,
                 eta_max=0.1,
                 T_up=0,
                 gamma=1.0,
                 last_epoch=-1,
                 plateau_patience=10,
                 plateau_factor=0.5,
                 min_lr=1e-6,
                 early_stop_patience=3):
        
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected non-negative integer T_up, but got {}".format(T_up))

        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_up = T_up
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_i = T_0
        self.T_cur = last_epoch
        self.cycle = 0
        self.gamma = gamma
        self.min_lr = min_lr

        # Plateau 관련
        self.best_val_loss = float('inf')
        self.plateau_counter = 0
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor

        # Early stopping 관련
        self.stall_cycles = 0
        self.early_stop_patience = early_stop_patience

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            # warm-up phase
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                    for base_lr in self.base_lrs]
        else:
            # cosine decay phase
            return [base_lr + (self.eta_max - base_lr) *
                    (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None, val_loss=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur -= self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        # warm-up 구간이 아닐 때만 plateau 감지
        if val_loss is not None and self.T_cur >= self.T_up:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.plateau_counter = 0
                self.stall_cycles = 0  # 개선되었으면 early stop 카운터도 초기화
            else:
                self.plateau_counter += 1
                if self.plateau_counter >= self.plateau_patience:
                    old_eta_max = self.eta_max
                    self.eta_max = max(self.eta_max * self.plateau_factor, self.min_lr)
                    self.plateau_counter = 0
                    self.best_val_loss = float('inf')
                    self.stall_cycles += 1
                    print(f"[LR Scheduler] Plateau detected. eta_max reduced from {old_eta_max:.2e} to {self.eta_max:.2e}")
        
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def should_early_stop(self):
        return self.stall_cycles >= self.early_stop_patience
    
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, base_lr, max_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / self.warmup_epochs
            return [self.base_lr + factor * (self.max_lr - self.base_lr) for _ in self.base_lrs]
        else:
            return [self.max_lr for _ in self.base_lrs]