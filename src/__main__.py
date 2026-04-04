import argparse
import warnings
import torch

from .graph_model import ExercisePredictor
from .testing import test_model
from .training import create_dataloaders, create_test_dataloaders, train_video_classifier


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(
        description='Process video frames for training or testing.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for the dataloader')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the csv files')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='Number of hidden dimensions')
    parser.add_argument('--num_layers', type=int,
                        default=512, help='Number of LSTM layers')
    parser.add_argument('--num_frames', type=int, default=60,
                        help='Number of frames to process')
    parser.add_argument('--test_path', type=str,
                        default='default', help='Path to the test json files')
    parser.add_argument('--checkpoint_path', type=str,
                        default='checkpoints', help='Path to the model checkpoint for testing')

    args = parser.parse_args()

    folder_name = args.checkpoint_path + "/" + str(args.num_layers) + \
        "-" + str(args.hidden_dim) + "/"

    print(f"Testing on {device} with model from {folder_name}")

    # Create checkpoint directory if it doesn't exist
    import os
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if args.test_path != 'default':
        # Load label maps from the training phase
        import json
        with open(folder_name + 'label_maps.json', 'r') as f:
            label_maps = json.load(f)

        test_loader = create_test_dataloaders(
            path=args.test_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_maps=label_maps
        )

        # Create and train model
        model = ExercisePredictor(output_size=len(label_maps.keys()),
                                  input_dim=3,
                                  num_nodes=33,
                                  embedding_dim=int(args.hidden_dim),
                                  lstm_hidden=int(args.hidden_dim),
                                  num_layers=int(args.num_layers))

        test_accuracy, report = test_model(
            model,
            f"{folder_name}best_model.pth",
            test_loader,
            device,
            len(label_maps.keys()),
            label_maps,
            hidden_dim=int(args.hidden_dim),
        )

        print(f'Test Accuracy: {test_accuracy:.2f}%')
        print(report)

        return

    train_loader, val_loader, test_loader, label_maps = create_dataloaders(
        path=args.path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # save label maps for later use in testing as a json file
    import json
    with open(folder_name + 'label_maps.json', 'w') as f:
        json.dump(label_maps, f)

    # Create and train model
    model = ExercisePredictor(output_size=len(label_maps.keys()),
                              input_dim=3,
                              num_nodes=33,
                              embedding_dim=int(args.hidden_dim),
                              lstm_hidden=int(args.hidden_dim),
                              num_layers=int(args.num_layers))

    train_video_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(label_maps.keys()),
        num_epochs=args.epochs,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir=folder_name,
        metrics_dir=folder_name + "metrics/",
        label_maps=label_maps,
        patience=25
    )

    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_accuracy, report = test_model(
        model,
        f"{folder_name}/best_model.pth",
        test_loader,
        device,
        len(label_maps.keys()),
        label_maps,
        hidden_dim=int(args.hidden_dim),
    )

    print(f'Test Accuracy: {test_accuracy:.6f}%')
    print(report)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
