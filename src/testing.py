import torch
from sklearn.metrics import (classification_report)
from torch import nn
from tqdm import tqdm
import numpy as np

from .metrics import AdvancedMetricsTracker


def test_model(model, path, test_loader, device, num_layers=2, hidden_dim=1024):

    # Load best model
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model = model.to(torch.float32)
    model.eval()

    criterion = nn.MSELoss()

    # Test loop
    val_losses = []

    with torch.no_grad():
        for videos in tqdm(test_loader):
            videos = videos.to(device)

            for frame_until in range(15, videos.shape[1]):
                h = torch.zeros(
                    num_layers, videos.shape[0], hidden_dim).to(device)
                c = torch.zeros(
                    num_layers, videos.shape[0], hidden_dim).to(device)

                outputs = model(videos[:, :frame_until, :, :])
                loss = criterion(
                    outputs, videos[:, frame_until, :, :].reshape(1, 99))
                val_losses.append(loss.item())

    # Calculate average, median, and std of validation losses
    avg_val_loss = np.mean(val_losses)
    median_val_loss = np.median(val_losses)
    std_val_loss = np.std(val_losses)

    # Save validation loss statistics to a JSON file
    val_loss_stats = {
        'average': avg_val_loss,
        'median': median_val_loss,
        'std_dev': std_val_loss
    }

    return val_loss_stats
