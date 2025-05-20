import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.csv_path = kwargs.get('csv_path', "")
        self.num_frames = kwargs.get('num_frames', 30)
        self.samples = []
        self.class_names = set()
        # Scan all json files and build index
        for fname in os.listdir(self.csv_path):
            if fname.endswith('.json'):
                class_name = fname.split(',')[0]
                self.class_names.add(class_name)
                self.samples.append((os.path.join(self.csv_path, fname), class_name))
        self.class_names = sorted(list(self.class_names))
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_path, class_name = self.samples[idx]
        with open(json_path, 'r') as f:
            pose_sequence = json.load(f)
        arr = np.array([[[kp['x'], kp['y'], kp['z']] for kp in pose] for pose in pose_sequence], dtype=np.float32)
        arr = arr.reshape(arr.shape[0], -1)  # Flatten the keypoints
        if arr.shape[0] > self.num_frames:
            indices = np.linspace(0, arr.shape[0] - 1, self.num_frames, dtype=int)
            arr = arr[indices]
        elif arr.shape[0] < self.num_frames:
            pad = np.zeros((self.num_frames - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        label = self.class_to_idx[class_name]
        return torch.from_numpy(arr), label
    
    def _get_label_map(self):
        return self.class_to_idx
