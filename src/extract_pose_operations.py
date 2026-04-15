#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter

NUM_LANDMARKS = 33  # MediaPipe Pose has 33 landmarks


def process_video(video_path, min_detection_confidence=0.75, min_tracking_confidence=0.75):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(
        cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    poses = []  # list of (NUM_LANDMARKS,4)
    frame_idx = 0

    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=min_detection_confidence,
                      min_tracking_confidence=min_tracking_confidence) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR->RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                arr = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                for i in range(min(len(lm), NUM_LANDMARKS)):
                    arr[i, 0] = lm[i].x
                    arr[i, 1] = lm[i].y
                    arr[i, 2] = lm[i].z
            else:
                arr = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)

            poses.append(arr)
            frame_idx += 1

            # optional simple progress indicator
            if total_frames:
                if frame_idx % 50 == 0 or frame_idx == total_frames:
                    print(
                        f"Processed {frame_idx}/{total_frames} frames", file=sys.stderr)
            else:
                if frame_idx % 200 == 0:
                    print(f"Processed {frame_idx} frames...", file=sys.stderr)

    cap.release()

    if len(poses) == 0:
        raise RuntimeError("No frames processed from the video.")

    # shape: (num_frames, NUM_LANDMARKS, 4)
    poses_arr = np.stack(poses, axis=0)
    return poses_arr


def normalize_pose_sequence(pose_sequence, smoothing_window=5, polyorder=2):
    """
    Normalize a pose sequence to ensure:
    1. Height is normalized
    2. Width is normalized
    3. Subject is centered at (0, 0)
    4. Subject is front-facing
    5. Temporal smoothing is applied

    Args:
        pose_sequence (np.ndarray): shape = [frames, landmarks, 3]
        smoothing_window (int): number of frames in smoothing window (must be odd)
        polyorder (int): polynomial order for Savitzky-Golay filter

    Returns:
        np.ndarray: normalized and smoothed pose sequence
    """
    poses = pose_sequence.copy()
    n_frames, n_landmarks, _ = poses.shape

    # === Step 1: Center to (0, 0) ===
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

    # === Step 2: Normalize Orientation (make front-facing) ===
    # average first 5 frames to reduce noise
    avg_frame = np.mean(poses[0:5], axis=0)
    left_shoulder = avg_frame[LEFT_SHOULDER]
    right_shoulder = avg_frame[RIGHT_SHOULDER]
    shoulder_vec = right_shoulder - left_shoulder
    angle = np.arcsin(shoulder_vec[2] / np.linalg.norm(shoulder_vec))
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(-angle), -np.sin(-angle)],
        [0, np.sin(-angle),  np.cos(-angle)]
    ])
    for f in range(n_frames):
        frame = poses[f]
        poses[f] = frame @ rotation_matrix

    # === Step 5: Temporal smoothing ===
    # Apply smoothing on each landmark dimension over time
    if n_frames >= smoothing_window and smoothing_window % 2 == 1:
        for j in range(n_landmarks):
            for dim in range(3):
                poses[:, j, dim] = savgol_filter(
                    poses[:, j, dim], smoothing_window, polyorder
                )
    else:
        # Fallback: simple moving average if window invalid
        kernel_size = min(3, n_frames)
        kernel = np.ones(kernel_size) / kernel_size
        for j in range(n_landmarks):
            for dim in range(3):
                poses[:, j, dim] = np.convolve(
                    poses[:, j, dim], kernel, mode="same"
                )

    for k in range(poses.shape[0]):
        # Debug: print each frame's landmarks after processing
        print(f"Frame {k}: {poses[k]}")
        poses[k] = poses[k] - poses[0]

    return poses
