import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.samples = []
        # Add all nii file path after traversing all folders in directory
        self.class_to_idx = {}
        # Scan all nii files and build index
        for root, _, files in os.walk(self.path):
            for fname in files:
                if fname.endswith('.nii'):
                    self.samples.append(os.path.join(root, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        nii_path = self.samples[idx]
        
        # Load the .nii file with nibabel
        import nibabel as nib
        nii_img = nib.load(nii_path)
        arr = nii_img.get_fdata()
        # Convert to numpy array and ensure correct shape
        arr = np.array(arr, dtype=np.float32)
        # Apply transforms if provided
        # if self.transform is not None:
        #     arr = self.transform(arr)
        
        return torch.from_numpy(arr)
   
    def _get_label_map(self):
        return self.class_to_idx
