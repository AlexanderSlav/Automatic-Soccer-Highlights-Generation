class TrainConfig:
    def __init__(self):
        self.model_name = 'squeezenet'
        self.log_interval = 2
        self.train_batch_size = 32
        self.test_batch_size = 16
        self.num_workers = 0
        self.epochs = 100
        self.datapath = 'binary_classification_soccer_dataset'
        self.criterion = 'crossentropy'
        self.input_size = 224