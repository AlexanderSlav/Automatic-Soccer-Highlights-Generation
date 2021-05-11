import cv2
import argparse
from tqdm import tqdm
import os
from utils import *
from typing import List
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='directory containing mp4 files.')
    parser.add_argument('--feature_extractor_name', type=str, default="resnet")
    parser.add_argument('--feature_extractor_weights', type=str)
    parser.add_argument('--gt_files', type=str, default="gt_summaries/test",  help='file with gt summary')
    parser.add_argument('--output_h5', type=str, help='path to output dataset.')
    parser.add_argument('--included_labels', type=str, nargs='+', required=True)
    args = parser.parse_args()
    return args


def event_duration(user_summary, fps=None):
    durations = []
    prev = 0
    count = 0
    for score in user_summary:
        if score == 1 and prev == 0:
            count += 1
        if score == 1 and prev == 1:
            count += 1
        elif score == 0 and prev == 1:
            durations.append(count)
            count = 0
        prev = score
    if fps is not None:
        durations = [duration // fps for duration in durations]
    return durations


def mean_variance_durations(durations):
    durations = np.hstack(durations)
    return np.mean(durations), np.std(durations)


def upsample_scores_to_original_size(frame_scores, picks, n_frames):
    picks = np.asarray(picks, dtype=np.int32)
    assert frame_scores.shape == picks.shape

    # Get original frame scores from downsampled sequence
    user_summary = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(picks)):
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        user_summary[pos_lo:pos_hi] = frame_scores[i]
    return user_summary


def imagenet_to_summarization(n_frames, path_to_file: str, inlcude_labels: List[int]):
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    with open(path_to_file) as f:
        lines = [line.split() for line in f]
    assert len(lines) == n_frames
    for frame_idx, (frame_name, target) in enumerate(lines):
        if target in inlcude_labels:
            frame_scores[frame_idx] = 1
    return frame_scores


def create_dataset(args):
    dataset = {}
    feauture_extractor = FeautureExtractor(net_name=args.feature_extractor_name,
                                           weights_path=args.feature_extractor_weights)
    all_durations = []
    for idx, video_file in enumerate(sorted(os.listdir(args.video_dir))):
        summary_file = os.path.join(args.gt_files, f"{video_file.split('.')[0]}.txt")
        video = cv2.VideoCapture(os.path.join(args.video_dir, video_file))
        video_name = video_file
        fps = video.get(cv2.CAP_PROP_FPS)
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        picks = []
        features = []
        # read video frame by frame and get frame features from googlenet
        for frame_idx in tqdm(range(n_frames - 1)):
            success, frame = video.read()
            if success:
                if frame_idx % 15 == 0:
                    frame_feat = feauture_extractor(frame)
                    picks.append(frame_idx)
                    features.append(frame_feat)
        included_labels = args.included_labels
        gt_score = imagenet_to_summarization(len(picks), summary_file, included_labels)
        user_summary = upsample_scores_to_original_size(gt_score, picks, n_frames)
        all_durations.append(event_duration(user_summary))
        change_points, n_frame_per_seg = get_change_points(np.asarray(features, dtype=np.float32), n_frames, fps)
        dataset[f'video_{idx}'] = {}
        dataset[f'video_{idx}']["features"] = np.asarray(features, dtype=np.float32)
        dataset[f'video_{idx}']["gtscore"] = np.asarray(gt_score, dtype=np.float32)
        dataset[f'video_{idx}']["user_summary"] = np.asarray(user_summary, dtype=np.float32)
        dataset[f'video_{idx}']["change_points"] = np.asarray(change_points, dtype=np.int32)
        dataset[f'video_{idx}']["n_frame_per_seg"] = np.asarray(n_frame_per_seg, dtype=np.int32)
        dataset[f'video_{idx}']["n_frames"] = np.asarray(n_frames, dtype=np.int32)
        dataset[f'video_{idx}']["picks"] = np.asarray(picks, dtype=np.int32)
        dataset[f'video_{idx}']["video_name"] = video_name
    mean, variance = mean_variance_durations(all_durations)
    print(mean, variance)
    return dataset


def main():
    args = parse_args()
    # with h5py.File(args.output_h5, 'w') as h5_f:
    dataset = create_dataset(args)
    convert_to_h5_dataset(dataset, args.output_h5)


if __name__ == "__main__":
    main()

