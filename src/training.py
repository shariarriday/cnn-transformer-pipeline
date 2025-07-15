import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report)
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import pandas as pd
from tqdm import tqdm
import json

from .metrics import AdvancedMetricsTracker
from .pose_dataset import PoseDataset
from .pose_transforms import get_training_transforms, get_validation_transforms, get_test_transforms

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
       
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
       
        return self.early_stop
   
def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': val_loss,    
    }
   
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
   
    best_val_loss = 1000000
    if os.path.exists(os.path.join(checkpoint_dir, 'best_model.pth')):
        best_checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
        best_val_loss = best_checkpoint['best_val_loss']
   
    # Save best model
    if best_val_loss > val_loss:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)

def train_video_classifier(
    model, train_loader, val_loader,
    num_epochs=20,
    device='cuda',
    checkpoint_dir='checkpoints',
    metrics_dir='metrics',
    patience=4,
    min_delta=0.01
):
    """Enhanced training loop with advanced metrics and early stopping"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
   
    # Check for existing checkpoint
    latest_checkpoint = None
    if os.path.exists(os.path.join(checkpoint_dir, 'latest_checkpoint.pth')):
        latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

    # Load checkpoint if available
    start_epoch = 0
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
   
    # Initialize optimizers and schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
   
    # Gradient scaler for mixed precision
    scaler = GradScaler(device)
   
    # Main scheduler
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_epochs
    )

    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
   
    metrics = AdvancedMetricsTracker()
   
    # Enable gradient checkpointing
    model.train()
    model = model.to(device)
   
    for epoch in range(start_epoch, num_epochs):
       
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (videos) in enumerate(tqdm(train_loader)):
            videos = videos.to(device)
           
            optimizer.zero_grad()
            
            outputs = model(videos)
            loss = criterion(outputs, videos)
            loss.backward()
            optimizer.step()
           
            train_loss += loss.item()
       
        main_scheduler.step()
       
        # Calculate metrics and save checkpoints
        avg_train_loss = train_loss / len(train_loader)
       
        model.eval()
       
        val_loss = 0.0
       
        with torch.no_grad():
            for videos in tqdm(val_loader):
                videos = videos.to(device)
           
                optimizer.zero_grad()
                
                outputs = model(videos)
                loss = criterion(outputs, videos)
                loss.backward()
                optimizer.step()
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
       
        # Update metrics
        current_lr = main_scheduler.get_last_lr()[0]
        metrics.update_epoch_metrics(
            avg_train_loss, avg_val_loss,
            current_lr
        )
       
        # Save all plots
        metrics.plot_training_curves(
            save_path=os.path.join(metrics_dir, f'training_curves_epoch_{epoch+1}.png')
        )
       
        save_checkpoint(model, optimizer, main_scheduler, epoch, avg_val_loss, checkpoint_dir)
       
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
       
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.8f}')
        print('-' * 80)
       
    return model
   
def create_dataloaders(*args, **kwargs):
    """Create train and validation dataloaders from a directory of pose JSON files"""
    pose_dataset_train = PoseDataset(os.path.join(kwargs.get('path'), 'train'), get_training_transforms())
    pose_dataset_val = PoseDataset(os.path.join(kwargs.get('path'), 'val'), get_validation_transforms())

    train_loader = DataLoader(
        pose_dataset_train,
        batch_size=kwargs.get('batch_size', 4),
        shuffle=True,
        num_workers=kwargs.get('num_workers', 4)
    )

    val_loader = DataLoader(
        pose_dataset_val,
        batch_size=kwargs.get('batch_size', 4),
        shuffle=False,
        num_workers=kwargs.get('num_workers', 4)
    )

    return train_loader, val_loader
