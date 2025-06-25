# config.yaml
seed: 42
data_path: /app_data/home/jonghyun/jhworkspace/ViTPose-Cus/data/custom/merged_3dpw_mpi.npz
save_path: /app_data/home/jonghyun/jhworkspace/ViTPose-Cus/MLP/checkpoints/2st.pth
log_dir: /app_data/home/jonghyun/jhworkspace/ViTPose-Cus/MLP/runs/mlp_log

train_split: 0.8
batch_size: 64          # per GPU (Total = 64 x 3 = 192)
epochs: 150
lr: 0.0005              # = 5e-4

scheduler:
  T_0: 150
  T_mult: 1
  eta_max: 0.0005
  eta_min: 1e-6
  T_up: 10
  gamma: 0.5