import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np


class PoseDataset(Dataset):
    def __init__(self, path, class_name, transform=None, label_maps=None):
        self.path = path
        self.transform = transform
        self.samples = []
        self.class_name = class_name
        self.class_names = set()

        # Scan all json files and build index
        for fname in os.listdir(self.path):
            if fname.endswith('.json'):
                if class_name == fname.split('__')[0]:
                    self.samples.append(os.path.join(self.path, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_path = self.samples[idx]
        with open(json_path, 'r') as f:
            pose_sequence = json.load(f)
        pose_data = []
        for poses in pose_sequence:
            all_data = []
            for kp in poses:
                all_data.append(
                    np.array([kp[0], kp[1], kp[2]], dtype=np.float32).reshape(3, -1))
            if len(all_data) == 33:
                pose_data.append(all_data)
        arr = np.array(pose_data, dtype=np.float32)
        arr = arr.reshape(arr.shape[0], -1)

        # Apply transforms if provided
        if self.transform is not None:
            arr = self.transform(arr)

        # Reshape back to (num_frames, num_keypoints, 3)
        arr = arr.reshape(arr.shape[0], 33, 3)

        ret = torch.from_numpy(arr).nan_to_num_(nan=0.0).float()
        if torch.isnan(ret).any() or torch.isinf(ret).any():
            print(
                f"Warning: NaN or Inf detected in logits for sample {idx} ({json_path})")
        return ret

    def _get_label_map(self):
        return self.class_to_idx

    def _set_label_map(self, label_map):
        self.class_to_idx = label_map

    def _set_class_name(self, class_name):
        self.class_name = class_name
