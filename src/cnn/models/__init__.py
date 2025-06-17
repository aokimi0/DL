from .baseline import BaselineNet
from .resnet import ResNet18
from .se_resnet import SEResNet18
from .densenet import DenseNet121

def get_model(name: str):
    name = name.lower()
    if name == "baseline":
        return BaselineNet()
    elif name == "resnet18":
        return ResNet18()
    elif name == "se_resnet18":
        return SEResNet18()
    elif name == "densenet121":
        return DenseNet121()
    else:
        raise ValueError(f"Model {name} not recognized.") 