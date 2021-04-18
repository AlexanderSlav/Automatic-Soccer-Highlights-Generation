import argparse
import skvideo.io
from utils import test_transforms
import torch
from model_builder import ModelBuilder
from tqdm import tqdm
from loguru import logger
import numpy as np
import cv2
classes = ['Celebration', 'Game Moment']


def parse_args():
    parser = argparse.ArgumentParser(description='Classify input image')
    parser.add_argument('--input_video', type=str, help='path to input video',
                        default='/home/alexander/Downloads/full_game_2.mp4')
    parser.add_argument('--output_video', type=str, help='path to input video',
                        default='full_game_summary_2.mp4')
    parser.add_argument('--model_path', type=str, help='path to model',
                        default='wandb/latest-run/files/best_model.pth')
    parser.add_argument('--model_name', type=str, help='model name'
                                                       'could be "squeezenet" or "resnet"', default='squeezenet')
    parser.add_argument('--fps_count', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()


class VideoWriter:
    def __init__(self, input_video_path, output_path):
        self.output_path = output_path
        self.video = skvideo.io.vreader(input_video_path)
        self.videometadata = skvideo.io.ffprobe(input_video_path)
        self.fps = int(self.videometadata['video']['@avg_frame_rate'].split('/')[0])
        output_dict = {
            "-vcodec": "libx265",
            "-vf": "format=yuv420p",
            "-movflags": "+faststart",
            "-r": f"{self.fps}",
        }
        self.writer = skvideo.io.FFmpegWriter(
            self.output_path, outputdict=output_dict, inputdict={"-r": f"{self.fps}"},
        )

    def __call__(self, summary):
        for idx, frame in tqdm(enumerate(self.video)):
            if summary[idx]:
                self.writer.writeFrame(frame)
        self.writer.close()
        logger.info(f"Saved as {self.output_path}")


class SoccerSummarizator:
    def __init__(self, model, device, batch_size: int, offset_time_step: int = 10):
        self.model = model
        self.device = device
        self.offset_time_step = offset_time_step
        self.batch_size = batch_size
        self._batch = []
        self._batch_idxs = []

    def _clean_batch(self):
        self._batch = []
        self._batch_idxs = []

    def _get_video_data(self, videometadata):
        frame_rate = int(videometadata['video']['@r_frame_rate'].split('/')[0])
        length = int(videometadata['video']['@nb_frames'])
        logger.info(f"Video lenth: {length} frames...")
        every_x_frame = frame_rate // 2
        downsampled_length = length // every_x_frame
        logger.info(f"Video was downsampled to {downsampled_length} frames...")
        picks = [every_x_frame * i for i in range(downsampled_length)]
        picks.append(length)
        return picks, frame_rate, length

    def __call__(self, video=None, videometadata=None):
        picks, orig_frame_rate, orig_frames_number = self._get_video_data(videometadata=videometadata)
        self.summary = np.array([0] * orig_frames_number)
        for idx, frame in tqdm(enumerate(video)):
            if idx in picks:
                input_tensor = test_transforms(image=frame)['image'].to(self.device, dtype=torch.float).unsqueeze(0)
                self._batch.append(input_tensor)
                self._batch_idxs.append(idx)
                if len(self._batch) == self.batch_size:
                    batch = torch.cat(self._batch, 0)
                    output = self.model(batch)
                    output = [(idx, result) for idx, result in zip(self._batch_idxs, output)]
                    self.generate_summary(output, orig_frame_rate)
                    self._clean_batch()
        return self.summary

    def generate_summary(self, outputs, orig_frame_rate):
        for idx, output in outputs:
            # if frame is celebration frame
            if torch.argmax(output).item() == 0:
                # select shot boundaries to include in summary
                left = max(idx - self.offset_time_step * orig_frame_rate, 0)
                right = idx
                self.summary[left:right] = 1


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # read video and video metadata
    video = skvideo.io.vreader(args.input_video)
    videometadata = skvideo.io.ffprobe(args.input_video)

    # build model
    model = ModelBuilder(args.model_name).get_model()
    model.load_state_dict(torch.load(args.model_path))
    model.to(device).eval()

    # build summarizator object
    summarizator = SoccerSummarizator(model=model, device=device, batch_size=args.batch_size)
    logger.info("Processing the video...")
    summarizator(video, videometadata=videometadata)

    logger.info("Save summary...")
    print(np.count_nonzero(summarizator.summary))
    # save video
    writer = VideoWriter(args.input_video, args.output_video)
    writer(summarizator.summary)


if __name__ == "__main__":
    main()


