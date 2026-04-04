# CNN Transformer Pipeline

This project trains and evaluates an exercise-classification model from pose landmark sequences.

## Project Structure

```
├── src
│   ├── __main__.py         # CLI entry point for training/testing
│   ├── graph_model.py      # ExercisePredictor model
│   ├── pose_dataset.py     # Dataset loading
│   ├── pose_transforms.py  # Pose/sequence transforms
│   ├── training.py         # Training + dataloader utilities
│   ├── testing.py          # Evaluation utilities
│   ├── metrics.py          # Metrics helpers
│   └── model.py
├── checkpoints/            # Saved models and label maps
├── dataset/                # Training/test data
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Development Environment

This repository includes a VS Code Dev Container configuration at `.devcontainer/devcontainer.json`.

- Name: `CUDA with Python`
- Base image: `mcr.microsoft.com/devcontainers/python:3.11`
- GPU: enabled via `--gpus=all`
- Host requirement: GPU available
- On-create dependencies: installs `torch`, `torchvision`, `mediapipe`, `pandas`, `seaborn`, `opencv-python`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `tqdm`

To use it in VS Code:

1. Install Docker and the VS Code Dev Containers extension.
2. Open this repository in VS Code.
3. Run: `Dev Containers: Reopen in Container`.

Note: The current dev container mounts a host dataset path (`D:/Dataset`) into `dataset/`. Update the `mounts` value in `.devcontainer/devcontainer.json` for your machine if needed.

## Docker

There is no project-specific `Dockerfile` in this repository. If you want to run with Docker directly, you can use the same base image used by the dev container.

### Build-Free Docker Run (Base Image)

```bash
docker run --rm -it \
	--gpus all \
	-v "$PWD":/workspaces/cnn-transformer-pipeline \
	-w /workspaces/cnn-transformer-pipeline \
	mcr.microsoft.com/devcontainers/python:3.11 \
	bash -lc "pip install -r requirements.txt && python -m src --num_workers 4 --batch_size 1 --epochs 200 --path 'dataset/landmark_poses_correct' --num_layers 2 --hidden_dim 256 --checkpoint_path classification"
```

If you do not have an NVIDIA GPU runtime configured, remove `--gpus all`.

## Usage

Run from the repository root:

```bash
python -m src [OPTIONS]
```

### Command-Line Arguments

Arguments below are taken from `src/__main__.py`:

- `--path` (required): Path to the training data/json-csv source used by dataloaders.
- `--num_workers` (default: `4`): Number of dataloader workers.
- `--batch_size` (default: `4`): Batch size.
- `--epochs` (default: `500`): Number of training epochs.
- `--hidden_dim` (default: `1024`): Hidden/embedding size.
- `--num_layers` (default: `512`): Number of LSTM layers (as wired in model creation).
- `--num_frames` (default: `60`): Number of frames to process.
- `--checkpoint_path` (default: `checkpoints`): Base directory for checkpoint artifacts.
- `--test_path` (default: `default`): If set to a real path, runs testing flow using saved label maps and checkpoint.

### Training Example

```bash
python -m src --num_workers 4 --batch_size 1 --epochs 200 --path "dataset/landmark_poses_correct" --num_layers 2 --hidden_dim 256 --checkpoint_path classification
```

### Testing Example

```bash
python -m src --path "dataset/landmark_poses_correct" --test_path "dataset/landmark_poses_wrong" --num_layers 2 --hidden_dim 256 --checkpoint_path classification
```

## Notes

- Checkpoints and label maps are stored under: `<checkpoint_path>/<num_layers>-<hidden_dim>/`
- Testing expects `label_maps.json` and `best_model.pth` to exist in that folder.

## License

This project is licensed under the MIT License. See LICENSE for details.