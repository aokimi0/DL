from typing import List
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layer_sizes: List[int], dropout_p: float = 0.2):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
        
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x) 