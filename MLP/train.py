import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from architecture.MLP import MLP
from utills.dataloader import KeypointDataset
from utills.logger import set_seed, save_checkpoint
from utills.visualizer import Visualizer
from utills.scheduler import WarmupScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from tqdm import tqdm

# === Load Config ===
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# === Set Seed ===
set_seed(config["seed"])

# === Device ===
device = torch.device("cuda:0")

# === Dataset ===
dataset = KeypointDataset(config["data_path"])
train_len = int(len(dataset) * config["train_split"])
val_len = len(dataset) - train_len
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# === Model ===
model = MLP().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

# === Scheduler ===
# scheduler = CosineAnnealingWarmUpRestartsWithPlateau(
#     optimizer,
#     T_0=config["scheduler"]["T_0"],
#     T_mult=config["scheduler"]["T_mult"],
#     eta_max=config["scheduler"]["eta_max"],
#     T_up=config["scheduler"]["T_up"],
#     gamma=config["scheduler"]["gamma"],
#     plateau_patience=config["scheduler"]["plateau_patience"],
#     plateau_factor=config["scheduler"]["plateau_factor"],
#     min_lr=config["scheduler"]["eta_min"],
#     early_stop_patience=config["scheduler"]["early_stop_patience"]
# )

warmup_epochs = int(config["warmup_epochs"])
base_lr = 1e-6

scheduler = WarmupScheduler(
    optimizer=optimizer,
    warmup_epochs=warmup_epochs,
    base_lr=base_lr,
    max_lr=float(config["lr"])
)

plateau_scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=float(config["scheduler"]["factor"]),
    patience=int(config["scheduler"]["patience"]),
    threshold=float(config["scheduler"]["threshold"]),
    min_lr=float(config["scheduler"]["min_lr"]),
    verbose=True
)

# === TensorBoard Visualizer ===
visualizer = Visualizer(log_dir=config["log_dir"])

dummy_input = torch.randn(1, 30).to(device)  # input_dim=30 (15×2)
visualizer.add_graph(model, dummy_input)

# === Training Loop ===
best_loss = float("inf")
prev_train_loss = float("inf")
for epoch in range(config["epochs"]):
    start_time = time.time()
    model.train()
    total_loss = 0
    grad_norm = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", ncols=100)
    for batch in pbar:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # === ✅ Gradient Clipping ===
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # === ✅ Gradient Norm 계산
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / ((pbar.n + 1) * config["batch_size"])
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

    avg_train_loss = total_loss / len(train_loader.dataset)

    # === Validation ===
    model.eval()
    val_loss = 0
    pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", ncols=100)
    with torch.no_grad():
        for batch in pbar_val:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

            avg_val_step = val_loss / ((pbar_val.n + 1) * config["batch_size"])
            pbar_val.set_postfix({"Loss": f"{avg_val_step:.4f}"})

    avg_val_loss = val_loss / len(val_loader.dataset)

    # === Logging ===
    # === Logging ===
    end_time = time.time()
    epoch_time = end_time - start_time
    current_lr = optimizer.param_groups[0]['lr']

    arrow_val = "⬇️" if avg_val_loss < best_loss else ("➖" if abs(avg_val_loss - best_loss) < 1e-6 else "⬆️")
    arrow_train = "⬇️" if avg_train_loss < prev_train_loss else ("➖" if abs(avg_train_loss - prev_train_loss) < 1e-6 else "⬆️")
    is_best = avg_val_loss < best_loss
    best_loss = min(best_loss, avg_val_loss)

    print(f"📊 Epoch {epoch+1:03}/{config['epochs']} | "
      f"Train: {avg_train_loss:.4f} {arrow_train} | "
      f"Val: {avg_val_loss:.4f} {arrow_val} | "
      f"Grad: {grad_norm:.3f} | LR: {current_lr:.2e} "
      f"{'💾' if is_best else ''}")

    # === TensorBoard 로그
    visualizer.log_metrics({
        "Train_Loss": avg_train_loss,
        "Val_Loss": avg_val_loss,
        "Grad_Norm": grad_norm,
        "LR": current_lr,
    }, step=epoch)

    # === TensorBoard Logging ===
    visualizer.log_metrics({
        "Train_Loss": avg_train_loss,
        "Val_Loss": avg_val_loss,
        "Grad_Norm": grad_norm,
        "LR": current_lr,
    }, step=epoch)

    # === Save Best Model ===
    if is_best:
        save_checkpoint(model, config["save_path"])

    if epoch < warmup_epochs:
        scheduler.step(epoch)
    else:
        plateau_scheduler.step(avg_val_loss)

visualizer.close()