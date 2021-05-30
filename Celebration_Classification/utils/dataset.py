import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from collections import Counter
from typing import Any

np.random.seed(42)
INPUT_SIZE = 224
import numpy as np


def set_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)


train_transforms = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE),
    A.HorizontalFlip(p=0.6),
    A.ColorJitter(p=0.5),
    A.OneOf([
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1),
    ], p=0.8),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),

])
test_transforms = A.Compose([
    A.Resize(INPUT_SIZE, INPUT_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),

])


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def get_class_distrubution(images, classes, split: str = "train"):
    targets = [image[1] for image in images]
    class_distrib = dict(Counter(targets))
    values = list(class_distrib.values())
    # tick_label does the some work as plt.xticks()
    plt.bar(range(len(class_distrib)), values, tick_label=classes)
    plt.savefig(f'class_distribution_{split}.png')
    plt.close()


class AlbumentationsDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(AlbumentationsDataset, self).__init__(root, transform)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)
            image = image['image']
        return image, target

    def __len__(self):
        return len(self.samples)


def load_split_train_test(train_opts, valid_size=.2):
    set_seed()
    datadir = train_opts.datapath
    train = AlbumentationsDataset(datadir, transform=train_transforms)
    test = AlbumentationsDataset(datadir, transform=test_transforms)
    num_train = len(train)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_data = Subset(train, train_idx)
    test_data = Subset(test, test_idx)
    train_images = [img for idx, img in enumerate(train_data.dataset.imgs) if idx in train_data.indices]
    test_images = [img for idx, img in enumerate(test_data.dataset.imgs) if idx in test_data.indices]
    # For unbalanced dataset we create a weighted samplers
    # Not working for some reasons with Subset and SubsetRandomSampler (classes were manually balanced)
    # train_weights = make_weights_for_balanced_classes(train.imgs, len(train_data.dataset.classes))
    # train_weights = torch.DoubleTensor(train_weights)
    # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights),replacement=True)
    # test_weights = make_weights_for_balanced_classes(test.imgs, len(test_data.dataset.classes))
    # test_weights = torch.DoubleTensor(test_weights)
    # test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights), replacement=True)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=train_opts.train_batch_size,
                                              num_workers=train_opts.num_workers)
    get_class_distrubution(train_images, train_data.dataset.classes,  split="train")
    testloader = torch.utils.data.DataLoader(test_data, batch_size=train_opts.test_batch_size,
                                             num_workers=train_opts.num_workers)
    get_class_distrubution(test_images, test_data.dataset.classes, split="test")
    return trainloader, testloader