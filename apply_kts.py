from kts.cpd_auto import cpd_auto
import argparse
import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    h5in = h5py.File(args.dataset, 'r')
    for video_name, video_file in h5in.items():
        features = video_file['features'][...].astype(np.float32)
        picks = video_file['picks'][...].astype(np.float32)
        n_frames = video_file['n_frames'][...].astype(np.float32)
        seq_len = features.shape[0]
        kernel = np.matmul(features, features.T)
        print(kernel)
        # kernel /= kernel.max()
        # kernel = np.matmul(features, features.T)
        change_points, _ = cpd_auto(kernel, seq_len - 1, 1)
        change_points *= 15
        change_points = np.hstack((0, change_points, n_frames))
        begin_frames = change_points[:-1]
        end_frames = change_points[1:]
        change_points = np.vstack((begin_frames, end_frames - 1)).T
        n_frame_per_seg = end_frames - begin_frames
        print(change_points, n_frame_per_seg)


if __name__ == "__main__":
    main()