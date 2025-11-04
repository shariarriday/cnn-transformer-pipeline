import torch
from tqdm import tqdm
import torch.nn as nn

from .metrics import AdvancedMetricsTracker

def test_model(model, test_loader, device, num_layers=2, hidden_dim=1024):
   
    # Load best model
    checkpoint = torch.load('checkpoints/best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    avg_loss = 0.0
    losses = []
    criterion = nn.L1Loss()

    with torch.no_grad():
        for videos in tqdm(test_loader):
            test_loss = 0.0
            videos = videos.to(device)

            h = torch.zeros(num_layers, videos.shape[0], hidden_dim).to(device)
            c = torch.zeros(num_layers, videos.shape[0], hidden_dim).to(device)

            for frame_until in range(15, videos.shape[1]):
                outputs, h, c = model(videos[:, :frame_until, :], h, c)
                loss = criterion(outputs, videos[:, frame_until, :])
                test_loss += loss.item()
            
            losses.append(test_loss)

    avg_loss = sum(losses) / len(losses)

    # Calculate and display results
    print(f'\nTest Loss: {avg_loss:.6f}')

    return avg_loss