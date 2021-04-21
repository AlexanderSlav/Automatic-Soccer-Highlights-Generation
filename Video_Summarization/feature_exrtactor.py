from torchvision import models
import cv2
from torch import nn
import argparse
import numpy as np
from tqdm import tqdm
import os
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='directory containing mp4 files.')
    parser.add_argument('--output_h5', type=str, help='path to output dataset.')
    args = parser.parse_args()
    return args


def create_dataset(net, args):
    dataset = {}
    for idx, video_file in enumerate(sorted(os.listdir(args.video_dir))[:1]):
        video = cv2.VideoCapture(os.path.join(args.video_dir, video_file))
        video_name = video_file
        fps = video.get(cv2.CAP_PROP_FPS)
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        picks = []
        features = []
        gtscore = []
        user_summary = []
        # read video frame by frame and get frame features from googlenet
        for frame_idx in tqdm(range(n_frames - 1)):
            success, frame = video.read()
            if success:
                if frame_idx % 15 == 0:
                    label = frame_annotate(frame)
                    user_summary.append(label)
                    gtscore.append(label)
                    frame_feat = net(transform(Image.fromarray(frame)).cuda().unsqueeze(0))
                    picks.append(frame_idx)
                    features.append(frame_feat.squeeze().detach().cpu().numpy())
        change_points, n_frame_per_seg = get_change_points(np.asarray(features, dtype=np.float32), n_frames, fps)
        dataset[f'video_{idx}'] = {}
        dataset[f'video_{idx}']["features"] = np.asarray(features, dtype=np.float32)
        dataset[f'video_{idx}']["gtscore"] = np.asarray(gtscore, dtype=np.float32)
        dataset[f'video_{idx}']["change_points"] = np.asarray(change_points, dtype=np.int32)
        dataset[f'video_{idx}']["n_frame_per_seg"] = np.asarray(n_frame_per_seg, dtype=np.int32)
        dataset[f'video_{idx}']["n_frames"] = np.asarray(n_frame_per_seg, dtype=np.int32)
        dataset[f'video_{idx}']["picks"] = np.asarray(picks, dtype=np.int32)
        dataset[f'video_{idx}']["video_name"] = video_name
    return dataset


def main():
    args = parse_args()
    net = models.googlenet(pretrained=True).float().cuda()
    net.eval()
    feture_net = nn.Sequential(*list(net.children())[:-2])
    # with h5py.File(args.output_h5, 'w') as h5_f:
    dataset = create_dataset(feture_net, args)
    convert_to_h5_dataset(dataset, args.output_h5)

if __name__ == "__main__":
    main()

