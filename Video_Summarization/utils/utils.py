from torchvision import transforms,models
import cv2
from kts.cpd_auto import cpd_auto
from PIL import Image
import math
import h5py
import numpy as np
import skvideo.io
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
from PIL import Image


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

    def __call__(self, picked_idxs=None, summary=None):
        if summary is not None:
            for idx, frame in tqdm(enumerate(self.video)):
                if summary[idx]:
                    self.writer.writeFrame(frame)
        elif picked_idxs:
            for idx, frame in tqdm(enumerate(self.video)):
                if idx in picked_idxs:
                    self.writer.writeFrame(frame)
        self.writer.close()
        logger.info(f"Saved as {self.output_path}")


class Rescale(object):
    """Rescale a image to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def view_dataset(h5_path):
    data = h5py.File(h5_path)
    features = data['video_0']['features'][...]
    picks = data['video_0']['picks'][...]
    n_frames = data['video_0']['n_frames'][...]
    print(features, n_frames, picks)


def get_change_points(features, n_frames, fps):
    kernel = np.matmul(features, features.T)
    n = n_frames / fps
    m = int(math.ceil(n / 2.0))
    change_points, _ = cpd_auto(kernel, m, 1)
    change_points *= 15
    change_points = np.hstack((0, change_points, n_frames))
    begin_frames = change_points[:-1]
    end_frames = change_points[1:]
    change_points = np.vstack((begin_frames, end_frames - 1)).T
    n_frame_per_seg = end_frames - begin_frames
    return change_points, n_frame_per_seg


def convert_to_h5_dataset(dataset, output_dataset_name):
    f = h5py.File(output_dataset_name, 'w')
    # video_names is a list of strings containing the
    # name of a video, e.g. 'video_1', 'video_2'
    for name, data in dataset.items():
        f.create_dataset(name + '/features', data=data['features'])
        f.create_dataset(name + '/gtscore', data=data['gtscore'])
        f.create_dataset(name + '/user_summary', data=data['user_summary'])
        f.create_dataset(name + '/change_points', data=data['change_points'])
        f.create_dataset(name + '/n_frame_per_seg', data=data['n_frame_per_seg'])
        f.create_dataset(name + '/n_frames', data=data['n_frames'])
        f.create_dataset(name + '/picks', data=data['picks'])
        f.create_dataset(name + '/video_name', data=data['video_name'])
    f.close()


class FeautureExtractor:
    def __init__(self):
        net = models.googlenet(pretrained=True).float().cuda()
        net.eval()
        self.feture_net = nn.Sequential(*list(net.children())[:-2])

    @staticmethod
    def preprocessing(image):
        return transform(Image.fromarray(image))

    def forward(self, data):
        return self.feture_net(data.cuda().unsqueeze(0))

    def __call__(self, image: np.ndarray):
        preprocessed_data = self.preprocessing(image)
        features = self.forward(preprocessed_data)
        return features.squeeze().detach().cpu().numpy()
