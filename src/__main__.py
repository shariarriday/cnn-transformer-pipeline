import argparse
import warnings
import torch

from .model import HybridCNNTransformerModel
from .testing import test_model
from .training import create_dataloaders, train_video_classifier

def main():
    parser = argparse.ArgumentParser(description='Process video frames for training or testing.')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to process')
    parser.add_argument('--num_workers', type=int, default=4, help='Number workers for dataloader')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for the frames')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for the dataloader')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file or frames directory')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the csv file')
    parser.add_argument('--target', type=str, default='l2_pose', help='target name')

    args = parser.parse_args()
    
    train_loader, val_loader, label_weights, num_classes, test_loader, label_maps, label_reverse_maps = create_dataloaders(
    csv_file=args.csv_path,
    video_folder=args.video_path,
    num_frames=args.num_frames,
    image_size=args.input_size,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    target=args.target
    )

    # Create and train model
    model = HybridCNNTransformerModel(num_classes=num_classes[args.target+"_count"],
                                    feature_dim=2048,
                                    frame_samples=args.num_frames,
                                    transformer_outputs=1024,
                                    transformer_layers=4,
                                    transformer_heads=64,
                                    dropout=0.2)

    train_video_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes[args.target+"_count"],
        num_epochs=args.epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='checkpoints',
        label_maps=label_maps,
        label_reverse_maps=label_reverse_maps,
        label_weights=label_weights[args.target],
        target=args.target,
        patience=4
    )

    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_accuracy, test_report = test_model(
        model,
        test_loader,
        device,
        num_classes[args.target+"_count"],
        label_maps,
        label_reverse_maps,
        target=args.target
    )

    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(test_report)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()