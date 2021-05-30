class TrainConfig:
    def __init__(self):
        self.model_name = 'squeezenet'
        self.log_interval = 2
        self.train_batch_size = 32
        self.test_batch_size = 16
        self.num_workers = 0
        self.epochs = 10
        self.datapath = '/home/alexander/HSE_Stuff/Diploma/Datasets/merged_dataset/goals_only'
        self.criterion = 'crossentropy'
        self.input_size = 224
        self.class_number = 2