import argparse
import os
import numpy as np
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i_d", type=str, help="Input directory with frames to split",
                        default="/home/alexander/HSE_Stuff/Diploma/Datasets/merged_dataset/celebration_only/game_moment")
    parser.add_argument("-o_d", type=str, help="Output directory",
                        default="/home/alexander/HSE_Stuff/Diploma/Datasets/merged_dataset/game_moment_truncated")
    parser.add_argument("-frames_amount", type=int, help="frames_amount", default=350)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    files = os.listdir(args.i_d)
    if not os.path.isdir(args.o_d):
        os.mkdir(args.o_d)
        print(f'Created the following directory: {args.o_d}')

    subset = np.random.choice(np.array(files), args.frames_amount, replace=False)
    for file in subset:
        cmd = f"cp {os.path.join(args.i_d, file)} {os.path.join(args.o_d, file)}"
        subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()