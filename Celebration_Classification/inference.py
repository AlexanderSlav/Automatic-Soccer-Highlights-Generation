import argparse
import torch
from soccer_summarizator import SoccerSummarizator


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize input video')
    parser.add_argument('--input_video', type=str, help='path to input video',
                        default='/home/alexander/Downloads/test.mp4')
    parser.add_argument('--output_video', type=str, help='path to input video',
                        default='celeb_recognition_summary.mp4')
    parser.add_argument('--model_name', type=str, help='model name'
                                                       'could be "squeezenet" or "resnet"', default='resnet')
    parser.add_argument('--classification_type', type=str, default='goals_from_celebration')
    parser.add_argument('--fps_count', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build summarizator object
    summarizator = SoccerSummarizator(model_name=args.model_name,
                                      input_video_path=args.input_video,
                                      output_video_path=args.output_video,
                                      device=device, batch_size=args.batch_size,
                                      classification_type=args.classification_type)
    # create summary and save it
    summarizator()


if __name__ == "__main__":
    main()