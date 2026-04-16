import torch
import json
import argparse
import warnings
import numpy as np

import extract_pose_operations
from graph_model import ExercisePredictor


def inference(model, weight_path, json_path, data, device, label_maps):

    # Load best model
    checkpoint = torch.load(
        weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        videos = data.to(device)

        outputs = model(videos)

        print(outputs)

        _, predicted = outputs.max(1)

    # Write output to json file after mapping back to class names
    predicted_class = None
    for item in label_maps.items():
        if item[1] == predicted.item():
            predicted_class = item[0]
            break

    with open(json_path, 'w') as f:
        json.dump({'predicted_class': predicted_class}, f)


def process_input(video_path):

    landmark = extract_pose_operations.process_video(video_path)
    landmark = extract_pose_operations.normalize_pose_sequence(
        landmark, smoothing_window=5, polyorder=2)

    pose_data = []
    for poses in landmark:
        all_data = []
        for kp in poses:
            all_data.append(
                np.array([kp[0], kp[1], kp[2]], dtype=np.float32).reshape(3, -1))
        if len(all_data) == 33:
            pose_data.append(all_data)
    arr = np.array(pose_data, dtype=np.float32)
    arr = arr.reshape(arr.shape[0], -1)
    arr = arr.reshape(1, arr.shape[0], 33, 3)

    print(f"Processed input: {arr}")

    return torch.from_numpy(arr).nan_to_num_(nan=0.0).float()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(
        description='Process video for inference.')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='Path to the model weights')
    parser.add_argument('--label_path', type=str, required=True,
                        help='Path to the label json file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output json files')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to the video file')

    args = parser.parse_args()

    # Load label maps from the training phase
    import json
    with open(args.label_path, 'r') as f:
        label_maps = json.load(f)

    # Create and train model
    model = ExercisePredictor(output_size=len(label_maps.keys()),
                              input_dim=3,
                              num_nodes=33,
                              embedding_dim=64,
                              lstm_hidden=64,
                              num_layers=1)

    inference(
        model=model,
        weight_path=args.weight_path,
        json_path=args.output_path,
        data=process_input(args.video_path),
        device=device,
        label_maps=label_maps
    )


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
