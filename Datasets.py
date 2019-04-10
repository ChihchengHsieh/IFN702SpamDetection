import torch
from torch.utils.data import Dataset, DataLoader


class HSapm14Dataset(Dataset):
    def __init__(self, text_ordered, X_p, X_len, y):
        super(HSapm14Dataset, self).__init__()

        self.text_ordered = text_ordered
        self.X_p = X_p
        self.X_len = X_len
        self.y = y

    def __len__(self,):
        return len(self.y)

    def __getitem__(self, idx):
        return self.text_ordered[idx], self.X_p[idx], self.X_len[idx], self.y[idx]
