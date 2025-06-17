import torch.nn as nn
from torchvision.models import densenet121

def DenseNet121(num_classes=10):
    model = densenet121(weights=None, num_classes=1000)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    return model 