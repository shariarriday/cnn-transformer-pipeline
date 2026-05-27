import json
import argparse
import warnings
import torch

from .graph_model import LandmarkPredictor
from .testing import test_model
from .training import create_dataloaders, create_test_dataloaders, train_video_classifier

class_names = {'bicycle-crunch': 0,
               'chair-squats': 1,
               'clap-patterns': 2,
               'exaggerated-side-steps': 3,
               }


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

        for class_name in class_names.keys():
            # Load label maps from the training phase
            test_loader = create_test_dataloaders(
                path=args.test_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                class_name=class_name
            )

            # Create and train model
            model = LandmarkPredictor(embedding_dim=int(args.hidden_dim),
                                      lstm_hidden=int(args.hidden_dim),
                                      num_layers=int(args.num_layers))

            test_loss = test_model(
                model,
                f"{folder_name}{class_name}/best_model.pth",
                test_loader,
                device,
                hidden_dim=int(args.hidden_dim),
            )

            import json

            with open(f'{folder_name}/{class_name}_loss_stats.json', 'w') as f:
                json.dump(test_loss, f, indent=4)

            print(f'Test Loss: {test_loss["average"]:.6f}')

        return

    for class_name in class_names.keys():

        train_loader, val_loader, test_loader = create_dataloaders(
            class_name=class_name,
            path=args.path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Create and train model
        model = LandmarkPredictor(embedding_dim=int(args.hidden_dim),
                                  lstm_hidden=int(args.hidden_dim),
                                  num_layers=int(args.num_layers))

        train_video_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            checkpoint_dir=f'{folder_name}{class_name}',
            metrics_dir=f'{folder_name}{class_name}/metrics',
            patience=15
        )

        # Test the model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        val_loss_stats = test_model(
            model,
            f'{folder_name}{class_name}/best_model.pth',
            test_loader,
            device,
            hidden_dim=int(args.hidden_dim),
        )

        print(f'Test loss: {val_loss_stats["average"]:.6f}')

        import json

        with open(f'{folder_name}/{class_name}_val_loss_stats.json', 'w') as f:
            json.dump(val_loss_stats, f, indent=4)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
