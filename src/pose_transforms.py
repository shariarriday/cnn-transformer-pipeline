import numpy as np
import torch
import random
from typing import List, Tuple, Optional

class PoseSequenceTransform:
    """Data augmentation transforms for pose sequences"""
    
    def __init__(self, transforms: List[str] = None, **kwargs):
        """
        Initialize pose sequence transforms
        
        Args:
            transforms: List of transform names to apply
            **kwargs: Parameters for specific transforms
        """
        self.transforms = transforms or []
        self.params = kwargs
        
        # Default parameters
        self.noise_std = kwargs.get('noise_std', 0.01)
        self.translation_range = kwargs.get('translation_range', 0.05)
        self.temporal_jitter_ratio = kwargs.get('temporal_jitter_ratio', 0.1)
        self.dropout_prob = kwargs.get('dropout_prob', 0.1)
        self.flip_prob = kwargs.get('flip_prob', 0.5)
        
    def __call__(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Apply transforms to pose sequence
        
        Args:
            pose_sequence: Shape (num_frames, num_keypoints * 3)
        
        Returns:
            Transformed pose sequence
        """
        # Reshape to (num_frames, num_keypoints, 3)
        num_frames, features = pose_sequence.shape
        num_keypoints = features // 3
        sequence = pose_sequence.reshape(num_frames, num_keypoints, 3)
        
        for transform_name in self.transforms:
            if hasattr(self, f'_{transform_name}'):
                sequence = getattr(self, f'_{transform_name}')(sequence)
        
        # Reshape back to (num_frames, num_keypoints * 3)
        return sequence.reshape(num_frames, -1)
    
    def _add_noise(self, sequence: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to keypoints"""
        noise = np.random.normal(0, self.noise_std, sequence.shape).astype(np.float32)
        return sequence + noise
    
    def _translate(self, sequence: np.ndarray) -> np.ndarray:
        """Random translation"""
        translation = np.random.uniform(
            -self.translation_range, self.translation_range, (1, 1, 3)
        ).astype(np.float32)
        return sequence + translation
    
    def _horizontal_flip(self, sequence: np.ndarray) -> np.ndarray:
        """Horizontal flip (mirror) with probability"""
        if np.random.random() < self.flip_prob:
            # Flip X coordinates
            sequence_copy = sequence.copy()
            sequence_copy[:, :, 0] = -sequence_copy[:, :, 0]
            return sequence_copy
        return sequence
    
    def _temporal_jitter(self, sequence: np.ndarray) -> np.ndarray:
        """Temporal jittering - slightly shift keypoints in time"""
        num_frames = sequence.shape[0]
        jitter_frames = int(num_frames * self.temporal_jitter_ratio)
        
        if jitter_frames > 0:
            # Create small random shifts for each frame
            shifts = np.random.randint(-jitter_frames, jitter_frames + 1, num_frames)
            shifts = np.clip(shifts, -num_frames//4, num_frames//4)  # Limit shifts
            
            new_sequence = np.zeros_like(sequence)
            for i, shift in enumerate(shifts):
                src_idx = np.clip(i + shift, 0, num_frames - 1)
                new_sequence[i] = sequence[src_idx]
            
            return new_sequence
        
        return sequence
    
    def _keypoint_dropout(self, sequence: np.ndarray) -> np.ndarray:
        """Randomly set some keypoints to zero (simulate occlusion)"""
        mask = np.random.random(sequence.shape[:2]) > self.dropout_prob
        mask = np.expand_dims(mask, axis=2)  # Add feature dimension
        return sequence * mask
    
    def _temporal_subsample(self, sequence: np.ndarray) -> np.ndarray:
        """Randomly subsample and interpolate frames"""
        num_frames = sequence.shape[0]
        if num_frames > 4:
            # Randomly remove 10-20% of frames and interpolate
            keep_ratio = np.random.uniform(0.8, 0.9)
            keep_frames = int(num_frames * keep_ratio)
            
            # Select frames to keep
            indices = np.sort(np.random.choice(num_frames, keep_frames, replace=False))
            subsampled = sequence[indices]
            
            # Interpolate back to original length
            from scipy.interpolate import interp1d
            old_indices = np.linspace(0, 1, keep_frames)
            new_indices = np.linspace(0, 1, num_frames)
            
            interpolated = np.zeros_like(sequence)
            for kp_idx in range(sequence.shape[1]):
                for coord_idx in range(sequence.shape[2]):
                    f = interp1d(old_indices, subsampled[:, kp_idx, coord_idx], 
                               kind='linear', bounds_error=False, fill_value='extrapolate')
                    interpolated[:, kp_idx, coord_idx] = f(new_indices)
            
            return interpolated
        
        return sequence
    
    def _normalize_pose(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize pose to have consistent scale and position"""
        # Center the pose (subtract mean position)
        mean_pos = np.mean(sequence, axis=(0, 1), keepdims=True)
        centered = sequence - mean_pos
        
        # Scale to unit variance
        std = np.std(centered) + 1e-8
        normalized = centered / std
        
        return normalized


class PoseTransformCompose:
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            sequence = transform(sequence)
        return sequence


# Predefined transform configurations
def get_training_transforms(**kwargs) -> PoseSequenceTransform:
    """Get standard training transforms"""
    return PoseSequenceTransform(
        transforms=[
            'add_noise',
            'scale',
            'rotate',
            'translate',
            'horizontal_flip',
            'temporal_jitter',
            'keypoint_dropout'
        ],
        **kwargs
    )

def get_validation_transforms(**kwargs) -> PoseSequenceTransform:
    """Get minimal validation transforms (usually just normalization)"""
    return PoseSequenceTransform(
        transforms=['normalize_pose'],
        **kwargs
    )

def get_test_transforms(**kwargs) -> PoseSequenceTransform:
    """Get test transforms (same as validation)"""
    return get_validation_transforms(**kwargs)
