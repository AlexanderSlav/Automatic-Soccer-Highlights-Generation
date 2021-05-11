import argparse
import torch
import skvideo
from Celebration_Classification.model_builder import ModelBuilder
from Celebration_Classification.soccer_summarizator import SoccerSummarizator, VideoWriter
from loguru import logger
from helpers import init_helper, data_helper, vsumm_helper, bbox_helper
from pathlib import Path
import os
ORIGINAL_VIDEOS_PATH = "/home/alexander/HSE_Stuff/Diploma/Datasets/custom/videos/test"

import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Soccer Summarizator on dataset for DSNET')
    parser.add_argument('--model_name', type=str, help='model name'
                                                       'could be "squeezenet" or "resnet"', default='squeezenet')
    parser.add_argument('--model_weights', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--splits', type=str, required=True)
    return parser.parse_args()


def evaluate(summarizator, val_loader):
    stats = data_helper.AverageMeter('fscore', 'diversity')
    with torch.no_grad():
        for test_key, seq, _, cps, n_frames, nfps, picks, user_summary in val_loader:
            video_name = os.path.basename(test_key)
            video_path = os.path.join(ORIGINAL_VIDEOS_PATH, f"{video_name}.mp4")

            video = skvideo.io.vreader(video_path)
            videometadata = skvideo.io.ffprobe(video_path)

            summarizator(video, videometadata=videometadata)
            pred_summ = summarizator.summary
            eval_metric = 'max'
            fscore = vsumm_helper.get_summ_f1score(
                pred_summ, user_summary, eval_metric)
            stats.update(fscore=fscore)
            # save video
            writer = VideoWriter(video_path, f"summaries/{video_name}_goals_summmary.mp4")
            writer(summarizator.summary)

    return stats.fscore, stats.diversity


def main():
    args = parse_args()
    logger.add(f"binary_classification_{args.model_name}_{args.splits.split('/')[-1]}.log")

    # build model
    model = ModelBuilder(args.model_name, num_classes=2).get_model()
    model.load_state_dict(torch.load(args.model_weights))
    model = nn.Sequential(*list(model.children())[:-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    summarizator = SoccerSummarizator(model=model, device=device, batch_size=args.batch_size, binary_closing=False)
    split_path = Path(args.splits)
    splits = data_helper.load_yaml(split_path)
    stats = data_helper.AverageMeter('fscore', 'diversity')
    for split_idx, split in enumerate(splits):
        val_set = data_helper.VideoDataset(split['test_keys'])
        val_loader = data_helper.DataLoader(val_set, shuffle=False)
        fscore, diversity = evaluate(summarizator, val_loader)
        logger.info(f"F-score:{round(fscore, 2)} Diversity:{round(diversity, 2)} on {split_idx} split")
        stats.update(fscore=fscore, diversity=diversity)
    logger.info(f'{split_path.stem}: diversity: {stats.diversity:.4f}, '
                f'F-score: {stats.fscore:.4f}')


if __name__ == "__main__":
    main()