import torch
from torchvision.transforms import v2 as transforms
import numpy as np
import random

class VideoTransform:
    def __init__(self, num_frames=16, input_size=224, mode='train'):
        self.mode = mode
        self.input_size = input_size
        self.num_frames = num_frames
        
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomApply([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomGrayscale(p=0.3),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    transforms.RandomInvert(p=0.3),
                    transforms.RandomRotation(degrees=30)
                ]),
                transforms.RandomResizedCrop(self.input_size, scale=(0.85, 1.0)),
                transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def temporal_crop(self, frames):
        if len(frames) <= self.num_frames:
            return frames
        start = random.randint(0, len(frames) - self.num_frames)
        return frames[start:start + self.num_frames]
    
    def speed_perturbation(self, frames, speed_range=(0.8, 1.2)):
        if len(frames) <= 1:
            return frames
        speed = random.uniform(*speed_range)
        num_frames = len(frames)
        new_num_frames = int(num_frames * speed)
        if new_num_frames <= 0:
            return frames
        indices = np.linspace(0, num_frames-1, new_num_frames).astype(int)
        return [frames[i] for i in indices]
    
    def frame_dropout(self, frames, p=0.1):
        if len(frames) <= self.num_frames:
            return frames
        mask = np.random.rand(len(frames)) > p
        if not np.any(mask):
            mask[random.randint(0, len(frames)-1)] = True
        return [f for f, m in zip(frames, mask) if m]
    
    def __call__(self, frames):
        if self.mode == 'train':
            frames = self.temporal_crop(frames)
            frames = self.speed_perturbation(frames)
            frames = self.frame_dropout(frames)
        
            frames = np.stack(frames)
            
        else:
            indices = np.linspace(0, len(frames)-1, self.num_frames).astype(int)
            frames = [frames[i] for i in indices]
            
            frames = np.stack(frames)
        
        frames = torch.from_numpy(frames)
        frames = frames.permute(0, 3, 1, 2)
        frames = [self.transform(frame) for frame in frames]

        frame_adjusment = self.num_frames - len(frames)
        if frame_adjusment > 0:
            for i in range(frame_adjusment):
                frames.append(frames[-1])
        
        return torch.stack(frames)