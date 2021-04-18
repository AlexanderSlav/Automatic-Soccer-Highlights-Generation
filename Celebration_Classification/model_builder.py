from torchvision import models
from torch import nn


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ModelBuilder:
    def __init__(self, model_name: str, num_classes: int = 2,
                 feature_extract: bool = True, use_pretrained: bool = True):
        self.model_name = model_name
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
        self.num_classes = num_classes

    def get_model(self):
        if self.model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, self.feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=self.use_pretrained)
            set_parameter_requires_grad(model_ft, self.feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = self.num_classes

        return model_ft