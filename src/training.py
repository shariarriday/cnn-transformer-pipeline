import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report)
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
import numpy as np
from torch.nn.utils import clip_grad_norm_

from .metrics import AdvancedMetricsTracker
from .pose_dataset import PoseDataset


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=7, min_delta=0.001):
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
        best_checkpoint = torch.load(
            os.path.join(checkpoint_dir, 'best_model.pth'))
        best_val_loss = best_checkpoint['best_val_loss']

    # Save best model
    if best_val_loss > val_loss:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)


def train_video_classifier(
    model, train_loader, val_loader,
    num_epochs=20,
    device='cuda',
    checkpoint_dir='ViT/checkpoints',
    metrics_dir='ViT/metrics',
    patience=4,
    min_delta=0.001,
    num_layers=1,
    hidden_dim=1024,
):
    """Enhanced training loop with advanced metrics and early stopping"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Check for existing checkpoint
    latest_checkpoint = None
    if os.path.exists(os.path.join(checkpoint_dir, 'latest_checkpoint.pth')):
        latest_checkpoint = os.path.join(
            checkpoint_dir, 'latest_checkpoint.pth')

    # Load checkpoint if available
    start_epoch = 0
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Initialize optimizers and schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)

    # Gradient scaler for mixed precision
    scaler = GradScaler(device)

    torch.autograd.set_detect_anomaly(True)

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

            start = np.random.randint(15, videos.shape[1] - 1)
            skip = np.random.randint(2, 5)

            optimizer.zero_grad()

            for frame_until in range(start, videos.shape[1], skip):
                h = torch.zeros(
                    num_layers, videos.shape[0], hidden_dim).to(device)
                c = torch.zeros(
                    num_layers, videos.shape[0], hidden_dim).to(device)
                outputs = model(videos[:, :frame_until, :, :])
                loss = criterion(
                    outputs, videos[:, frame_until, :, :].reshape(1, 99))

                train_loss += loss.item()

                loss.backward()

            optimizer.step()

        main_scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for videos in tqdm(val_loader):
                videos = videos.to(device)

                start = np.random.randint(15, videos.shape[1] - 1)
                skip = np.random.randint(2, 5)

                for frame_until in range(start, videos.shape[1], skip):
                    h = torch.zeros(
                        num_layers, videos.shape[0], hidden_dim).to(device)
                    c = torch.zeros(
                        num_layers, videos.shape[0], hidden_dim).to(device)

                    outputs = model(videos[:, :frame_until, :, :])
                    loss = criterion(
                        outputs, videos[:, frame_until, :, :].reshape(1, 99))
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
            save_path=os.path.join(
                metrics_dir, f'training_curves_epoch_{epoch+1}.png')
        )

        save_checkpoint(model, optimizer, main_scheduler,
                        epoch, avg_val_loss, checkpoint_dir)

        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.10f}')
        print(f'Val Loss: {avg_val_loss:.10f}')
        print(f'Learning Rate: {current_lr:.10f}')
        print('\nClassification Report:')
        print('-' * 80)

    return model


def create_dataloaders(*args, **kwargs):
    """Create train and validation dataloaders from a directory of pose JSON files"""
    pose_dataset = PoseDataset(kwargs.get('path'), kwargs.get('class_name'))

    train_ds, valid_ds, test_ds = torch.utils.data.random_split(
        pose_dataset,
        [0.85, 0.1, 0.05],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=kwargs.get('batch_size', 4),
        shuffle=True,
        num_workers=kwargs.get('num_workers', 4)
    )

    val_loader = DataLoader(
        valid_ds,
        batch_size=kwargs.get('batch_size', 4),
        shuffle=False,
        num_workers=kwargs.get('num_workers', 4)
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=kwargs.get('batch_size', 4),
        shuffle=False,
        num_workers=kwargs.get('num_workers', 4)
    )

    return train_loader, val_loader, test_loader


def create_test_dataloaders(*args, **kwargs):
    """Create train and validation dataloaders from a directory of pose JSON files"""
    pose_dataset = PoseDataset(kwargs.get('path'), kwargs.get('class_name'))

    test_loader = DataLoader(
        pose_dataset,
        batch_size=kwargs.get('batch_size', 4),
        shuffle=False,
        num_workers=kwargs.get('num_workers', 4)
    )

    return test_loader
