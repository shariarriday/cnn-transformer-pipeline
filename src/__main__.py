import argparse
import numpy as np
import torch

from .model import HybridCNNTransformerModel
from .testing import test_model
from .training import create_dataloaders, train_video_classifier

def main():
    parser = argparse.ArgumentParser(description='Process video frames for training or testing.')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to process')
    parser.add_argument('--num_workers', type=int, default=4, help='Number workers for dataloader')
    parser.add_argument('--input_size', type=int, default=224, help='Input size for the frames')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the video file or frames directory')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the csv file')
    parser.add_argument('--target', type=str, default='l2_pose', help='target name')

    args = parser.parse_args()
    
    train_loader, val_loader, num_classes, _, label_maps, label_reverse_maps = create_dataloaders(
    csv_file=args.csv_path,
    video_folder=args.video_path,
    num_frames=16,
    image_size=224,
    batch_size=4,
    num_workers=0
    )

    # Create and train model
    model = HybridCNNTransformerModel(num_classes=num_classes[args.target+"_count"],
                                    feature_dim=512,
                                    frame_samples=16,
                                    transformer_outputs=256,
                                    transformer_layers=4,
                                    transformer_heads=16,
                                    dropout=0.25)

    model = train_video_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes[args.target+"_count"],
        num_epochs=25,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir='checkpoints',
        label_maps=label_maps,
        label_reverse_maps=label_reverse_maps,
        target=args.target,
        patience=4
    )

    # Create test dataloader
    _, _, num_classes, test_loader, label_maps, label_reverse_maps = create_dataloaders(
        csv_file=args.csv_path,
        video_folder=args.video_path,
        frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=2
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
    main()