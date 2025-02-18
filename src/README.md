# cnn-transformer-app/cnn-transformer-app/README.md

# CNN Transformer Video Processing Application

This project implements a video processing application using a combination of Convolutional Neural Networks (CNN) and Transformers. The application is designed to perform various transformations on video frames, which can be useful for tasks such as pose detection.

## Project Structure

```
├── src
│   ├── __main__.py             # Entry point for the application
│   ├── video_transform.py      # Contains the VideoTransform class for frame transformations
│   ├── video_dataset.py        # Contains the VideoDataset class for data loading and preprocessing
│   ├── model.py                # Defines the CNN and Transformer architectures
│   ├── training.py             # Implements training routines and optimizer settings
│   ├── testing.py              # Contains testing routines and evaluation scripts
│   └── metrics.py              # Provides functions for computing evaluation metrics
├── requirements.txt            # Lists the project dependencies
└── README.md                   # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd cnn-transformer-app
pip install -r requirements.txt
```

## Usage

You can run the application from the command line. The entry point is located in `src/__main__.py`. You can specify the mode (train or test) and provide the necessary video frames for processing.

### Command Line Arguments

- `--mode`: Specify the mode of operation. Options are `train` or `test`. Default is `train`.
- `--video_path`: Path to the video file or frames directory.
- `--csv_path`: Path to the csv file.
- `--num_frames`: Number of frames to process.
- `--num_workers`: Number workers for dataloader.
- `--input_size`: Input size for the frames.
- `--target`: Target name.

### Example

To run the application in training mode with a specified input and output path, use the following command:

```bash
python -m src --mode train --input path/to/input --output path/to/output
```

## Video Transformations

The `VideoTransform` class includes several methods for transforming video frames:

- **temporal_crop**: Randomly crops a sequence of frames.
- **temporal_jitter**: Adds jitter by randomly sampling frames.
- **speed_perturbation**: Simulates different playback speeds.
- **frame_dropout**: Randomly drops frames from the sequence.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.