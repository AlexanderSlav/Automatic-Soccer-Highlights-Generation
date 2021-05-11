import argparse
import math
import random
from pathlib import Path

import h5py
import yaml


def make_random_splits(keys, num_test, num_splits):
    splits = []
    for _ in range(num_splits):
        random.shuffle(keys)
        test_keys = keys[:num_test]
        train_keys = list(set(keys) - set(test_keys))
        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys
        })
    return splits


def make_cross_val_splits(keys, num_videos, num_test):
    random.shuffle(keys)
    splits = []
    for i in range(0, num_videos, num_test):
        test_keys = keys[i: i + num_test]
        train_keys = list(set(keys) - set(test_keys))
        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys
        })
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to h5 dataset')
    parser.add_argument('--save-path', type=str, default='splits/resnet_goals_train.yml',
                        help='Path to save generated splits')
    parser.add_argument('--num-splits', type=int, default=4,
                        help='How many splits to generate')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                        help='Percentage of training data')
    parser.add_argument('--method', type=str, default='cross',
                        choices=['random', 'cross'],
                        help='Random selection or cross validation')
    args = parser.parse_args()

    dataset = h5py.File(args.dataset, 'r')
    keys = list(dataset.keys())
    keys = [str(Path(args.dataset) / key) for key in keys]
    if args.num_splits == 1:
        num_test = len(keys)
        num_videos = len(keys)
    else:
        num_videos = len(keys)
        num_train = int(math.ceil(num_videos * args.train_ratio))
        num_test = num_videos - num_train

    if args.method == 'random':
        splits = make_random_splits(keys, num_test, args.num_splits)
    elif args.method == 'cross':
        splits = make_cross_val_splits(keys, num_videos, num_test)
    else:
        raise ValueError(f'Invalid method {args.method}')

    # save splits
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(splits, f)


if __name__ == '__main__':
    main()
