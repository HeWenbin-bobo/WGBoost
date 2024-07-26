import torch
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, x, y):
        self.feat = torch.tensor(x)
        self.label = torch.tensor(y)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.feat[idx, :], self.label[idx]
