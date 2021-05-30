import torch
import wandb
from torch import optim
from torch import nn
from utils.opts import TrainConfig
from model_builder import ModelBuilder
from trainer import Trainer
import numpy as np
import datetime
from loguru import logger

def set_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

wandb.init(project="Celebration_Classification")


def main(train_opts):
    time_now = datetime.datetime.now().strftime("%H:%M")
    dataset_name = train_args.datapath.split('/')[-1]
    run_name = f"{dataset_name}_{time_now}"
    logger.info(run_name)
    wandb.run.name = run_name
    wandb.save()
    set_seed()
    wandb.config.update(train_opts)
    # as the baseline, we will use squeezenet lightweight classification model
    model = ModelBuilder(train_opts.model_name,
                         num_classes=train_opts.class_number).get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)
    trainer = Trainer(train_args, model, criterion, optimizer_ft, wandb)
    wandb.watch(model)
    trainer.run_training()


if __name__ == "__main__":
    train_args = TrainConfig()
    main(train_args)