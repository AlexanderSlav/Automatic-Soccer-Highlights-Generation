import cv2
from torch import nn
import argparse
import numpy as np
from tqdm import tqdm
import os
from utils import *
import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='directory containing mp4 files.')
    parser.add_argument('--gt_summary_file', type=str, default="summaries.h5",  help='file with gt summary')
    parser.add_argument('--output_h5', type=str, help='path to output dataset.')
    args = parser.parse_args()
    return args


def create_dataset(net, args):
    dataset = {}
    feauture_extractor = FeautureExtractor()
    summaries = h5py.File(args.gt_summary_file, 'r')
    for idx, video_file in enumerate(sorted(os.listdir(args.video_dir))):
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
        change_points, n_frame_per_seg = get_change_points(np.asarray(features, dtype=np.float32), n_frames, fps)
        dataset[f'video_{idx}'] = {}
        dataset[f'video_{idx}']["features"] = np.asarray(features, dtype=np.float32)
        dataset[f'video_{idx}']["gtscore"] = summaries[f'video_{idx}']['gtscore'][...].astype(np.float32)
        dataset[f'video_{idx}']["user_summary"] = summaries[f'video_{idx}']['user_summary'][...].astype(np.float32)
        dataset[f'video_{idx}']["change_points"] = np.asarray(change_points, dtype=np.int32)
        dataset[f'video_{idx}']["n_frame_per_seg"] = np.asarray(n_frame_per_seg, dtype=np.int32)
        dataset[f'video_{idx}']["n_frames"] = np.asarray(n_frames, dtype=np.int32)
        dataset[f'video_{idx}']["picks"] = np.asarray(picks, dtype=np.int32)
        dataset[f'video_{idx}']["video_name"] = video_name
    return dataset


def main():
    args = parse_args()
    # with h5py.File(args.output_h5, 'w') as h5_f:
    dataset = create_dataset(args)
    convert_to_h5_dataset(dataset, args.output_h5)

if __name__ == "__main__":
    main()

