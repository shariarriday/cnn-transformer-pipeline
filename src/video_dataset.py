import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2

class VideoDataset(Dataset):
    def __init__(self, csv_file, video_folder, num_frames=16, train=True, transform=None, **kwargs):
        """
        Args:
            csv_file (str): Path to the CSV file.
            video_folder (str): Path to the folder containing videos.
            transform (callable, optional): Transform to apply to frames.
        """
        self.csv = csv_file
        self.data = pd.read_csv(csv_file)
        self.video_folder = video_folder
        self.transform = transform
        self.train = train
        self.num_frames = num_frames
        self.data = self.remove_sequence_ids_with_missing_video()
        if train:
            self.filter_all_train_data()
        else:
            self.filter_all_test_data()
        _, _, self.label_reverse_maps = self._get_label_map()
        
    
    def _get_label_map(self):
        """
        Get label maps for each of the targets.
        """
        train_data = pd.read_csv(self.csv)

        label_counts = {
            "l1_pose_count": train_data["l1_pose_id"].nunique(),
            "l2_pose_count": train_data["l2_pose_id"].nunique(),
            "l3_pose_count": train_data["l3_pose_id"].nunique()
        }
        label_maps = {
            "l1_pose": dict(zip(train_data["l1_pose_id"], train_data["l1_pose"])),
            "l2_pose": dict(zip(train_data["l2_pose_id"], train_data["l2_pose"])),
            "l3_pose": dict(zip(train_data["l3_pose_id"], train_data["l3_pose"]))
        }
        label_reverse_maps = {
            "l1_pose": {pose_id: idx for idx, pose_id in enumerate(sorted(train_data["l1_pose_id"].unique()))},
            "l2_pose": {pose_id: idx for idx, pose_id in enumerate(sorted(train_data["l2_pose_id"].unique()))},
            "l3_pose": {pose_id: idx for idx, pose_id in enumerate(sorted(train_data["l3_pose_id"].unique()))}
        }

        return label_counts, label_maps, label_reverse_maps
        
    def remove_sequence_ids_with_missing_video(self):
        """
        Remove sequence IDs with missing video files.
        """
        valid_rows = []
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            video_path = os.path.join(self.video_folder, f"{row['sequence_id']}.mp4")
            if not os.path.exists(video_path):
                continue
            valid_rows.append(row)
        df_filtered = pd.DataFrame(valid_rows)
        df_filtered.reset_index(drop=True, inplace=True)
        return df_filtered
        
    def filter_all_test_data(self):
        """
        Filter all test data.
        """
        self.data = self.data[self.data["split"] == "test"]
        self.data.reset_index(drop=True, inplace=True)
        
    def filter_all_train_data(self):
        """
        Filter all train data.
        """
        self.data = self.data[self.data["split"] == "train"]
        self.data.reset_index(drop=True, inplace=True)
        
    def _load_video(self, video_path):
        """
        Load all frames from the video file.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        
        if len(frames) > 300:
            # sample 100 frames
            indices = np.linspace(0, len(frames) - 1, 300, dtype=int)
            frames = [frames[i] for i in indices]
        
        return np.stack(frames)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Ensure we find a valid video file
        row = self.data.iloc[idx]
        # Read the video
        frames = self._load_video(os.path.join(self.video_folder, f"{row['sequence_id']}.mp4"))

        if self.transform:
            frames = self.transform(frames)
        
        return frames, self.label_reverse_maps["l2_pose"][row["l2_pose_id"]]
    

class CurriculumVideoDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_stage = 0
        self.max_stages = 4
        self.frame_lengths = [4, 8, 12, 16]  # Increasing sequence lengths
        
    def update_curriculum(self, epoch):
        """Update curriculum stage based on epoch"""
        self.curriculum_stage = min(epoch // 5, self.max_stages - 1)
        self.num_frames = self.frame_lengths[self.curriculum_stage]
        
    def __getitem__(self, idx):
        frames, label = super().__getitem__(idx)
        return frames[:self.num_frames], label