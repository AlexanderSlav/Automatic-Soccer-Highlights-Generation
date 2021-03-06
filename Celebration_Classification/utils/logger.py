from loguru import logger

class Logger:
    def __init__(self, wandb, args, num_samples):
        self.wandb = wandb
        self.args = args
        self.num_samples = num_samples

    def batch_log(self, epoch, batch_idx, loss):
        if batch_idx % self.args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * self.args.train_batch_size}/{self.num_samples}] '
              f'\tLoss: {round(loss, 3)}')

    def epoch_log(self, accuracy, loss, stage, **kwargs):
        data = {f"{stage.capitalize()} Accuracy": accuracy[stage].avg,
            f"{stage.capitalize()} Loss": loss[stage].avg}
        data.update(kwargs)
        logger.info(f"Epoch: {data}")
        self.wandb.log(data)

    def final_accuracy(self, accuracy):
        logger.info(f"Per class Accuracy: {accuracy}")
