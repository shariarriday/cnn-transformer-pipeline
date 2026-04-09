import os
import shutil
import random
from collections import defaultdict


def train_test_split():
    # Define paths
    processed_folder = "dataset/landmark_poses_correct"
    train_folder = "train"
    val_folder = "val"
    test_folder = "test"

    # Create output directories if they don't exist
    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    # Get all files from Processed folder
    if not os.path.exists(processed_folder):
        print(f"Error: {processed_folder} directory not found")
        return

    files = [f for f in os.listdir(processed_folder) if os.path.isfile(
        os.path.join(processed_folder, f))]

    # Group files by class name (last part after splitting by comma)
    class_files = defaultdict(list)
    for file in files:
        # Extract class name from filename (last part after comma split)
        class_name = file.split(',')[-1].split('.')[0]  # Remove file extension
        class_files[class_name.split('__')[0]].append(file)

    # Split files for each class
    for class_name, file_list in class_files.items():
        # random.shuffle(file_list)
        total_files = len(file_list)
        train_count = int(0.8 * total_files)
        val_count = int(0.1 * total_files)

        train_files = file_list[:train_count]
        val_files = file_list[train_count:train_count + val_count]
        test_files = file_list[train_count + val_count:]

        # Copy files to respective folders
        for file in train_files:
            shutil.copy2(os.path.join(processed_folder, file),
                         os.path.join(train_folder, file))

        for file in val_files:
            shutil.copy2(os.path.join(processed_folder, file),
                         os.path.join(val_folder, file))

        for file in test_files:
            shutil.copy2(os.path.join(processed_folder, file),
                         os.path.join(test_folder, file))

        print(
            f"Class {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


if __name__ == "__main__":
    train_test_split()
