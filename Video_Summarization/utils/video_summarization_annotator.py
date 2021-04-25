import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import warnings
from loguru import logger
import json
import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='directory containing truncated mp4 files.')
    parser.add_argument('--picks_file', type=str, help='file contatining picks for every video')
    parser.add_argument('--out_summary_file_name', type=str, default="summaries.h5",
                        help='out  file containing summaries.')
    args = parser.parse_args()
    return args


class VideoAnnotator:
    def __init__(self, time_step: int = 10, video_picks: dict = None):
        self.time_step = time_step
        self.video_picks = video_picks

    def create_user_summarry(self, gtscore, video_name):
        gtscore = np.asarray(gtscore, dtype=np.int32)
        picks = np.asarray(self.video_picks[video_name]["picks"], dtype=np.int32)
        # Get original frame scores from downsampled sequence
        n_frames = self.video_picks[video_name]["n_frames"]
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        for i in range(len(picks)):
            pos_lo = picks[i]
            pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
            frame_scores[pos_lo:pos_hi] = gtscore[i]
        return frame_scores

    def __call__(self, video_path, video_name):
        video = cv2.VideoCapture(video_path)
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        gtscore = np.zeros(n_frames, dtype=np.float32)
        frame_counter = 0
        for frame_idx in tqdm(range(n_frames - 1)):
            success, frame = video.read()
            if success:
                frame_counter += 1
                cv2.imshow(f'frame_{frame_idx}', frame)
                if frame_counter == self.time_step:
                    frame_counter = 0
                    logger.info("Provide binary label for previous shot")
                    key = None
                    while key != ord('0') and key != ord('1'):
                        key = cv2.waitKey(0)
                        if key == ord('0'):
                            label = 0
                        elif key == ord('1'):
                            label = 1
                        else:
                            warnings.warn("You should set label: 0 for non-include frames and 1 for include frames")
                    pos_lo = max(frame_idx - self.time_step, 0)
                    pos_hi = frame_idx
                    gtscore[pos_lo:pos_hi] = label
                    logger.info(gtscore)
                else:
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        user_summary = self.create_user_summarry(gtscore, video_name)
        return gtscore, user_summary


def main():
    args = parse_args()
    with open(args.picks_file) as picks_file:
        video_info = json.load(picks_file)
    annotator = VideoAnnotator(time_step=5, video_picks=video_info)
    summaries = {}
    for idx, video_file in enumerate(sorted(os.listdir(args.video_dir))):
        video_path = os.path.join(args.video_dir, video_file)
        gtscore, user_summary = annotator(video_path, video_file)
        summaries[f'video_{idx}'] = {}
        summaries[f'video_{idx}']["gtscore"] = np.asarray(gtscore, dtype=np.float32)
        summaries[f'video_{idx}']["user_summary"] = np.asarray(user_summary, dtype=np.float32)
    f = h5py.File(args.out_summary_file_name, 'w')
    for name, data in summaries.items():
        f.create_dataset(name + '/gtscore', data=data['gtscore'])
        f.create_dataset(name + '/user_summary', data=data['user_summary'])
    f.close()


if __name__ == "__main__":
    main()

