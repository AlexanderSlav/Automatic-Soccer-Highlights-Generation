import argparse
import skvideo.io
from utils import test_transforms
import torch
from model_builder import ModelBuilder
from tqdm import tqdm
from loguru import logger
import numpy as np
import scipy.ndimage as nd

class_to_idxs = {
    "celebration": 0,
    "goals": 1,
    "celebration_goals": 1
}

# Set threshold for celebration event to 120 frames
CELEBRATION_DURATION_TRESHOLD = 120
# Set mean duration for goals event to 150 frames (data from exploratory annotation data analysis)
GOALS_MEAN_DURATION = 140


def parse_args():
    parser = argparse.ArgumentParser(description='Classify input image')
    parser.add_argument('--input_video', type=str, help='path to input video',
                        default='/home/alexander/HSE_Stuff/Diploma/Datasets/custom/videos/test/video_0.mp4')
    parser.add_argument('--output_video', type=str, help='path to input video',
                        default='celeb_recognition_summary.mp4')
    parser.add_argument('--model_path', type=str, help='path to model',
                        default='wandb/celebration-only-balanced-resnet/files/best_model.pth')
    parser.add_argument('--model_name', type=str, help='model name'
                                                       'could be "squeezenet" or "resnet"', default='resnet')
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
    def __init__(self, model, device, batch_size: int, binary_closing: bool = False,
                 classification_type: str = "celebration"):
        """
        classification_type: str  could be "celebration" or "goal"
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self._batch = []
        self._batch_idxs = []
        self.binary_closing = binary_closing
        self.classification_type = classification_type

    def _clean_batch(self):
        self._batch = []
        self._batch_idxs = []

    def _get_video_data(self, videometadata):
        fps_data = videometadata['video']['@avg_frame_rate'].split('/')[0]
        frame_rate = int(fps_data) if len(fps_data) == 2 else int(fps_data[:2])
        try:
            length = int(videometadata['video']['@nb_frames'])
        except:
            length = int(float(videometadata['video']["@duration"]) * frame_rate)
        logger.info(f"Video lenth: {length} frames...")
        every_x_frame = frame_rate // 2
        downsampled_length = length // every_x_frame
        logger.info(f"Video was downsampled to {downsampled_length} frames...")
        picks = [every_x_frame * i for i in range(downsampled_length)]
        picks.append(length)
        return picks, frame_rate, length

    def upsample_scores_to_original_size(self, picks, n_frames):
        picks = np.asarray(picks, dtype=np.int32)
        assert self.summary.shape == picks.shape

        # Get original frame scores from downsampled sequence
        user_summary = np.zeros(n_frames, dtype=np.float32)
        for i in range(len(picks)):
            pos_lo = picks[i]
            pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
            user_summary[pos_lo:pos_hi] = self.summary[i]
        return user_summary

    def __call__(self, video=None, videometadata=None):
        picks, orig_frame_rate, orig_frames_number = self._get_video_data(videometadata=videometadata)
        self.summary = np.array([0] * len(picks))
        for idx, frame in tqdm(enumerate(video)):
            if idx in picks:
                input_tensor = test_transforms(image=frame)['image'].to(self.device, dtype=torch.float).unsqueeze(0)
                self._batch.append(input_tensor)
                self._batch_idxs.append(idx)
                if len(self._batch) == self.batch_size:
                    batch = torch.cat(self._batch, 0)
                    output = self.model(batch)
                    output = [(idx, result) for idx, result in zip(self._batch_idxs, output)]
                    self.generate_summary(output, picks)
                    self._clean_batch()
        if self.binary_closing:
            self.summary = nd.binary_closing(self.summary).astype(np.int32)
        self.summary = self.upsample_scores_to_original_size(picks, orig_frames_number)
        start_idxs_durations = self.get_event_start_idxs_durations()
        if self.classification_type == "celebration":
            self.convert_to_goals_only(start_idxs_durations)
        elif self.classification_type == "goal":
            self.goal_event_duration_check(start_idxs_durations)
        return self.summary

    def get_event_start_idxs_durations(self):
        """
        Only for celebration classification case when we want to get goals
        event from detected celebration event by offset n frames back
        """
        durations = []
        start_idxs = []
        prev = 0
        count = 0
        for idx, score in enumerate(self.summary):
            if score == 1 and prev == 0:
                count += 1
                start_idxs.append(idx)
            if score == 1 and prev == 1:
                count += 1
            elif score == 0 and prev == 1:
                durations.append(count)
                count = 0
            prev = score
        return dict(zip(start_idxs, durations))

    def convert_to_goals_only(self, event_start_idxs_durations):
        for start_idx, duration in event_start_idxs_durations.items():
            if duration >= CELEBRATION_DURATION_TRESHOLD:
                self.summary[start_idx-GOALS_MEAN_DURATION:start_idx] = 1
            self.summary[start_idx:start_idx + duration] = 0

    def goal_event_duration_check(self, event_start_idxs_durations):
        for start_idx, duration in event_start_idxs_durations.items():
            if duration < GOALS_MEAN_DURATION:
                self.summary[start_idx:start_idx + duration] = 0

    def generate_summary(self, outputs, picks):
        for idx, output in outputs:
            if torch.argmax(output).item() == class_to_idxs[self.classification_type]:
                try:
                    insert_idx = picks.index(idx)
                    self.summary[insert_idx] = 1
                except:
                    print(f'Not in list {idx}')


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


