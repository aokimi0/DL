import torch
import torch.nn as nn
from typing import Tuple


class NNLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(NNLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, input_tensor: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # input_tensor: (batch_size, seq_len, input_size)
        output, hidden = self.lstm(input_tensor, hidden)
        # output: (batch_size, seq_len, hidden_size)
        # We only need the output of the last time step
        last_output = output[:, -1, :]
        output = self.fc(last_output)
        output = self.log_softmax(output)
        return output, hidden

    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        hidden_state = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
        cell_state = torch.zeros(1, batch_size, self.lstm.hidden_size).to(device)
        return hidden_state, cell_state 