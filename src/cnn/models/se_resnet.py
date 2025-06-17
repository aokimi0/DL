import torch
import torch.nn as nn
from .resnet import ResNet, BasicBlock
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(BasicBlock):
    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__(in_planes, planes, stride)
        self.se = SEBlock(planes * self.expansion, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out = self.se(out)

        out += residual
        out = F.relu(out)

        return out

def SEResNet18(num_classes=10):
    return ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes) 