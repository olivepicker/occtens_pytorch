import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from einops import rearrange

class SceneDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        path = d.path
        f = np.load(os.path.join('data', path))
        sem = f['semantics']
        mask_lidar = f['mask_lidar']
        mask_camera = f['mask_camera']
        
        sem, mask_lidar, mask_camera = map(lambda t: rearrange(t, 'x y z -> z y x'), (sem, mask_lidar, mask_camera))
        valid_mask = (mask_lidar > 0) | (mask_camera > 0)
        sem[~valid_mask] = 18
        out = {
            'semantic': torch.tensor(sem).long(),
            'mask_lidar': torch.tensor(mask_lidar).long(),
            'mask_camera': torch.tensor(mask_camera).long()
        }

        return out