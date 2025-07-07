from .mnist import MnistSimpleModel, ConvModel
from .resnet import ResNet18

def create_mnist_model(model_type = "simple"):
    if model_type == "simple":
        return MnistSimpleModel()
    elif model_type == "conv":
        return ConvModel()
    elif model_type == "resnet18":
        return ResNet18(in_channels = 1)
    else:
        raise ValueError("In create_mnist_model(): model_type is not approriate")

def create_cifar_model(model_type = "resnet18"):
    if model_type == "resnet18":
        return ResNet18(in_channels = 3)
    elif model_type == "conv":
        return ConvModel(in_channels = 3, H = 32, W = 32)
    else:
        raise ValueError("In create_cifar_model(): model_type is not approriate")