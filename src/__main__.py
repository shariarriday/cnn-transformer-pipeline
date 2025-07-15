import argparse
import warnings
import torch

from .model import Conv3DAutoencoder
from .testing import test_model
from .training import create_dataloaders, train_video_classifier

def main():
    parser = argparse.ArgumentParser(description='Process video frames for training or testing.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the dataloader')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--path', type=str, required=True, help='Path to the csv files')
    parser.add_argument('--num_frames', type=int, default=60, help='Number of frames to process')

    args = parser.parse_args()
   
    train_loader, val_loader = create_dataloaders(
    path=args.path,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    num_frames=args.num_frames,
    )

    # Create and train model
    model = Conv3DAutoencoder()

    train_video_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='checkpoints',
        patience=25
    )

    # Test the model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # test_accuracy, test_report = test_model(
    #     model,
    #     test_loader,
    #     device,
    #     len(label_maps.keys()),
    #     label_maps,
    #     num_layers=int(args.num_layers) * 2,  # LSTM is bidirectional
    #     hidden_dim=int(args.hidden_dim),
    # )

    # print(f'Test Accuracy: {test_accuracy:.2f}%')
    # print(test_report)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
