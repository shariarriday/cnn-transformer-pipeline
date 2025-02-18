import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report)
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, SWALR
import pandas as pd
from tqdm import tqdm
import json

from .metrics import AdvancedMetricsTracker
from .video_dataset import CurriculumVideoDataset
from .video_transform import VideoTransform

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
    
def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # Save best model
    if best_val_acc == checkpoint['best_val_acc']:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)

def train_video_classifier(
    model, train_loader, val_loader, 
    num_epochs=20,
    num_classes=1,
    device='cuda', 
    checkpoint_dir='checkpoints',
    metrics_dir='metrics',
    patience=4,
    min_delta=0.01,
    target='l2_pose',
    label_maps=None,
    label_reverse_maps=None
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
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
    
    # Initialize optimizers and schedulers
    optimizer = torch.optim.Adafactor(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(device)
    
    # Stochastic Weight Averaging
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-2)
    swa_start = num_epochs // 4
    
    # Warmup scheduler
    warmup_epochs = 5
    if start_epoch < warmup_epochs:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
    
    # Main scheduler
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    # Initialize metrics tracker
    classes = []
    for i in range(num_classes):
        for item in label_reverse_maps[target].items():
            if item[1] == i:
                classes.append(label_maps[target][item[0]])
    
    metrics = AdvancedMetricsTracker(num_classes=num_classes, classes=classes)
    best_val_acc = 0.0
    
    # Enable gradient checkpointing
    model.train()
    if hasattr(model, 'cnn_backbone'):
        model.cnn_backbone.gradient_checkpointing_enable()
    
    model = model.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        # Update curriculum
        if isinstance(train_loader.dataset, CurriculumVideoDataset):
            train_loader.dataset.update_curriculum(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        for batch_idx, (videos, labels) in enumerate(tqdm(train_loader)):
            videos, labels = videos.to(device), labels.to(device)
            # Mixed precision training
            with autocast(device):
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update SWA model if in SWA phase
            if epoch >= swa_start:
                swa_model.update_parameters(model)
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Update schedulers
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
            
        if epoch >= swa_start:
            swa_scheduler.step()
        
        # Calculate metrics and save checkpoints
        train_accuracy = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase with SWA model if in SWA phase
        eval_model = swa_model if epoch >= swa_start else model
        eval_model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for videos, labels in tqdm(val_loader):
                videos, labels = videos.to(device), labels.to(device)
                outputs = eval_model(videos)
                probabilities = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        val_accuracy = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update metrics
        current_lr = swa_scheduler.get_last_lr()[0]
        metrics.update_epoch_metrics(
            avg_train_loss, avg_val_loss, 
            train_accuracy, val_accuracy,
            current_lr
        )
        metrics.update_predictions(
            torch.tensor(all_predictions), 
            torch.tensor(all_labels),
            torch.tensor(all_probabilities)
        )
        
        # Save all plots
        metrics.plot_training_curves(
            save_path=os.path.join(metrics_dir, f'training_curves_epoch_{epoch+1}.png')
        )
        metrics.plot_confusion_matrix(
            save_path=os.path.join(metrics_dir, f'confusion_matrix_epoch_{epoch+1}.png')
        )
        metrics.plot_roc_curves(
            save_path=os.path.join(metrics_dir, f'roc_curves_epoch_{epoch+1}.png')
        )
        metrics.plot_precision_recall_curves(
            save_path=os.path.join(metrics_dir, f'pr_curves_epoch_{epoch+1}.png')
        )
        
        # Save classification report
        report = classification_report(
            metrics.metrics['epoch_labels'],
            metrics.metrics['epoch_predictions'],
            labels=list(range(num_classes)),
            target_names=metrics.classes,
            output_dict=True
        )
        
        # Save report as JSON
        with open(os.path.join(metrics_dir, f'classification_report_epoch_{epoch+1}.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        # Update best validation accuracy and save checkpoint
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_checkpoint(model, optimizer, swa_scheduler, epoch, best_val_acc, checkpoint_dir)
            
        
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        print('\nClassification Report:')
        print(pd.DataFrame(report).transpose())
        print('-' * 80)
        
        return model
    
def create_dataloaders(*args, **kwargs):
	"""Create train and validation dataloaders from a directory of videos"""
	video_dataset_train = CurriculumVideoDataset(*args, train=True, transform=VideoTransform(kwargs.get('num_frames', 16), kwargs.get('image_size', 224), mode='train'), **kwargs)
	video_dataset_test = CurriculumVideoDataset(*args, train=False, transform=VideoTransform(kwargs.get('num_frames', 16), kwargs.get('image_size', 224), mode='test'), **kwargs)

	train_ds, valid_ds = torch.utils.data.random_split(
		video_dataset_train, 
		[0.85, 0.15]
	)

	# Use PyTorch's DataLoader with memory pinning
	train_loader = DataLoader(
		train_ds,
		batch_size=kwargs.get('batch_size', 8),
		shuffle=True,
		pin_memory=True,
		num_workers=kwargs.get('num_workers', 4)
	)

	val_loader = DataLoader(
		valid_ds,
		batch_size=kwargs.get('batch_size', 8),
		shuffle=False,
		pin_memory=True,
		num_workers=kwargs.get('num_workers', 4)
	)

	test_loader = DataLoader(
		video_dataset_test,
		batch_size=kwargs.get('batch_size', 8),
		shuffle=False,
		pin_memory=True,
		num_workers=kwargs.get('num_workers', 4)
	)

	label_counts, label_maps, label_reverse_maps = video_dataset_train._get_label_map()
	return train_loader, val_loader, label_counts, test_loader, label_maps, label_reverse_maps