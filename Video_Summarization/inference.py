import argparse
from pathlib import Path
import numpy as np
import torch
from anchor_free.dsnet_af import DSNetAF
from helpers import data_helper, vsumm_helper, bbox_helper
from modules.model_zoo import get_model
from loguru import logger
import cv2
from tqdm import tqdm
from utils.utils import *


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--device', type=str, default='cuda',
                        choices=('cuda', 'cpu'))
    parser.add_argument('--video', type=str)
    # common model config
    parser.add_argument('--base-model', type=str, default='attention',
                        choices=['attention', 'lstm', 'linear', 'bilstm',
                                 'gcn'])
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1024)
    parser.add_argument('--num-hidden', type=int, default=128)

    parser.add_argument('--nms-thresh', type=float, default=0.5)

    args = parser.parse_args()
    return args


def feature_extraction(video, n_frames):
    feauture_extractor = FeautureExtractor()
    features = []
    picks = []
    # read video frame by frame and get frame features from googlenet
    for frame_idx in tqdm(range(n_frames - 1)):
        success, frame = video.read()
        if success:
            if frame_idx % 15 == 0:
                picks.append(frame_idx)
                frame_feat = feauture_extractor(frame)
                features.append(frame_feat)
    features = np.asarray(features, dtype=np.float32)
    return features, picks


def run():
    args = parse_args()
    logger.info(vars(args))
    writer = VideoWriter(args.video, 'test.mp4')
    # create model and load model weights
    model = DSNetAF(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, num_head=args.num_head)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
    model.eval().to(args.device)

    # read video and video info
    video = cv2.VideoCapture(args.video)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    # downsample video and extract frames
    features, picks = feature_extraction(video, n_frames)


    # get kts change points
    change_points, n_frame_per_seg = get_change_points(features, n_frames, fps)

    seq_len = len(features)
    features = torch.from_numpy(features).unsqueeze(0)
    with torch.no_grad():
        pred_cls, pred_bboxes = model.predict(features.to(args.device))
        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, args.nms_thresh)
        n_frame_per_seg = np.asarray(n_frame_per_seg, dtype=np.int32)
        change_points = np.asarray(change_points, dtype=np.int32)
        picks = np.asarray(picks, dtype=np.int32)
        pred_summ = vsumm_helper.bbox2summary(
            seq_len, pred_cls, pred_bboxes, change_points,
            n_frames, n_frame_per_seg, picks)
        writer(summary=pred_summ)


if __name__ == '__main__':
    run()