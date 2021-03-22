from torchvision import transforms, models
import torch
import cv2
from torch import nn
from PIL import Image
import argparse
import h5py
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='directory containing mp4 files.')
    parser.add_argument('--output_h5', type=str, help='path to output dataset.')
    args = parser.parse_args()
    return args


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
    features = data['video_1']['features'][()]
    picks = data['video_1']['picks'][()]
    n_frames = data['video_1']['n_frames'][()]
    print(features, n_frames, picks)


def video2features(net, args, h5_file):
    video = cv2.VideoCapture(args.video_dir)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=length)
    ratio = 15
    downsample_rate = length // ratio
    print(downsample_rate)
    features = []
    success, frame = video.read()
    i = 0
    # read video frame by frame and get frame features from googlenet
    while success:
        pbar.update(1)
        if (i + 1) % ratio == 0:
            features.append(net(transform(Image.fromarray(frame)).cuda().unsqueeze(0)).squeeze().detach().cpu().numpy())
        i += 1
        success, frame = video.read()
    features = np.asarray(features)
    picks = [ratio * i for i in range(downsample_rate)]
    v_data = h5_file.create_group('video_1')
    v_data['features'] = features
    v_data['picks'] = picks
    v_data['n_frames'] = length


def main():
    args = parse_args()
    net = models.googlenet(pretrained=True).float().cuda()
    net.eval()
    feture_net = nn.Sequential(*list(net.children())[:])
    print(feture_net)
    # with h5py.File(args.output_h5, 'w') as h5_f:
    #     video2features(feture_net, args, h5_f)
    # view_dataset(args.output_h5)


if __name__ == "__main__":
    main()

