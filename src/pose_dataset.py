import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = []
        # Scan all nii files and build index
        for fname in os.listdir(self.path):
            if fname.endswith('.nii'):
                self.files.append(os.path.join(self.path, fname))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        nii_path = self.files[idx]
        nii_image = nib.load(nii_path)
        arr = nii_image.get_fdata()
        arr = arr.astype(np.float32)
        arr = np.nan_to_num(arr)
        return torch.from_numpy(arr)
