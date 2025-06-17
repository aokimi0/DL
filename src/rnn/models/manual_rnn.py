import torch
import torch.nn as nn
from typing import Tuple


class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualRNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor: torch.Tensor, hidden_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.log_softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size) 