from utils import VideoWriter
import argparse
import os
import cv2
from tqdm import tqdm
import json
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='directory containing mp4 files.')
    parser.add_argument('--out_dir', type=str, help='directory containing truncated mp4 files.')
    args = parser.parse_args()
    return args


def downsample_videos(video_dir, out_dir):
    if out_dir is None:
        out_dir = os.getcwd()
    else:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
            logger.info(f'Created the following directory: {out_dir}')
    video_dir_picks = {}
    for idx, video_file in enumerate(sorted(os.listdir(video_dir))):
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(out_dir, video_file)
        writer = VideoWriter(video_path, output_path)
        video = cv2.VideoCapture(video_path)
        video_name = video_file
        picks = []
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_idx in tqdm(range(n_frames - 1)):
            success, frame = video.read()
            if success:
                if frame_idx % 15 == 0:
                    picks.append(frame_idx)
        video_dir_picks[video_name] = {"picks": picks,
                                       "n_frames": n_frames}
        writer(picks)
    output_picks_path = "video_picks.json"
    with open(output_picks_path, "w") as outfile:
        json.dump(video_dir_picks, outfile, indent=1, sort_keys=True)


def main():
    args = parse_args()
    downsample_videos(args.video_dir, args.out_dir)


if __name__ == "__main__":
    main()