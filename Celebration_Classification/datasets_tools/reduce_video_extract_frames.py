import os
from pathlib import Path
import argparse
import glob
from tqdm import tqdm
import subprocess
import cv2
import datetime
import math


class FrameExtractor:
    '''
    Class used for extracting frames from a video file.
    '''

    def __init__(self):
        pass

    def open_video(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        duration = self.n_frames / self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    def extract_frames(self, imgs_name, every_x_frame: int = 1,
                       dest_path=None, img_ext='.png'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f'Created the following directory: {dest_path}')

        frame_cnt = 0
        img_cnt = 0

        while self.vid_cap.isOpened():

            success, image = self.vid_cap.read()

            if not success:
                break

            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([imgs_name, '_', str(img_cnt), img_ext]))
                cv2.imwrite(img_path, image)
                img_cnt += 1

            frame_cnt += 1

        self.vid_cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i_d", type=str, help="Input directory with videos", default="../custom/videos/original")
    parser.add_argument("-o_f", type=str, help="Output directory with frames", default="../custom/frames")
    parser.add_argument("-every_x_frame", type=int, help="Every x frame", default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    frame_extractor = FrameExtractor()
    for file in tqdm(sorted(os.listdir(args.i_d))[:3]):
        input_file = os.path.join(args.i_d, file)
        frame_extractor.open_video(input_file)
        frame_extractor.extract_frames(imgs_name=file[:-4], every_x_frame=args.every_x_frame, dest_path=args.o_f)


if __name__ == "__main__":
    main()