import torch
import torch.nn as nn
from typing import Tuple


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, input_tensor: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        output, hidden = self.lstm(input_tensor, hidden)
        
        output = self.log_softmax(self.out(output))
        
        return output, hidden

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size),
        ) 