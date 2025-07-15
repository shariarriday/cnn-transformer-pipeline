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
        self.class_names = set()
        # Scan all json files and build index
        for fname in os.listdir(self.path):
            if fname.endswith('.json'):
                self.class_names.add(fname.split('__')[0])
                self.samples.append((os.path.join(self.path, fname), fname.split('__')[0]))
        self.class_names = sorted(list(self.class_names))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        print(self.class_to_idx)    
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_path, class_name = self.samples[idx]
        with open(json_path, 'r') as f:
            pose_sequence = json.load(f)
        pose_data = []
        for poses in pose_sequence:
            all_data = []
            for kp in poses:
                all_data.append(np.array([kp['x'], kp['y'], kp['z'],kp['visibility']], dtype=np.float32).reshape(4, -1))
            if len(all_data) == 33:        
                pose_data.append(all_data)
        arr = np.array(pose_data, dtype=np.float32)
        arr = arr.reshape(arr.shape[0], -1)
        
        # Apply transforms if provided
        if self.transform is not None:
            arr = self.transform(arr)
        
        label = self.class_to_idx[class_name]
        return torch.from_numpy(arr), label
   
    def _get_label_map(self):
        return self.class_to_idx
