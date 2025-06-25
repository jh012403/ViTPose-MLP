import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class KeypointDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.keypoints_2d = data['keypoints_2d']  # (N, 14, 2)
        self.keypoints_3d = data['keypoints_3d']  # (N, 14, 3)

        # Flatten for MLP input/output
        self.inputs = self.keypoints_2d.reshape(len(self.keypoints_2d), -1)  # (N, 28)
        self.targets = self.keypoints_3d.reshape(len(self.keypoints_3d), -1)  # (N, 42)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]).float(), torch.from_numpy(self.targets[idx]).float()


def get_dataloaders(npz_path, batch_size=64, split_ratio=0.8, num_workers=2):
    dataset = KeypointDataset(npz_path)
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
