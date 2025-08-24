
import torch
from torch.utils.data import Dataset, DataLoader
class RandomMaskDataset(Dataset):
    def __init__(self,n=64,in_ch=3,H=128,W=128): self.n=n; self.in_ch=in_ch; self.H=H; self.W=W
    def __len__(self): return self.n
    def __getitem__(self, idx): x = torch.rand(self.in_ch, self.H, self.W); y = (torch.rand(1, self.H, self.W)>0.5).float(); return x,y
def get_dataloaders(batch_size: int = 8, **kwargs):
    train = DataLoader(RandomMaskDataset(n=128), batch_size=batch_size, shuffle=True)
    val = DataLoader(RandomMaskDataset(n=32), batch_size=batch_size)
    test = DataLoader(RandomMaskDataset(n=32), batch_size=batch_size)
    return train, val, test
