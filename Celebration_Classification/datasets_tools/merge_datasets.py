import argparse
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i_d", type=str, help="Input directory with video frames",
                        default="/home/alexander/HSE_Stuff/Diploma/Datasets/imagenet_like_custom_dataset")
    parser.add_argument("-o_d", type=str, help="Output directory",
                        default="/home/alexander/HSE_Stuff/Diploma/Datasets/merged_dataset")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.isdir(args.o_d):
        os.mkdir(args.o_d)
        print(f'Created the following directory: {args.o_d}')
    for dirpath, _, filenames in os.walk(args.i_d):
        for filename in filenames:
            class_name = dirpath.split('/')[-1]
            class_dir = os.path.join(args.o_d, class_name)
            video_name = dirpath.split('/')[-2]
            if not os.path.isdir(class_dir):
                os.mkdir(class_dir)
                print(f'Created the class directory: {class_name}')
            out_file = os.path.join(class_dir, f"{video_name}_{filename}")
            cmd = f"cp {os.path.join(dirpath, filename)} {out_file}"
            subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    main()