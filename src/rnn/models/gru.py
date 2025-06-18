import torch
import torch.nn as nn
from typing import Tuple


class ManualGRU(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ManualGRU, self).__init__()
        self.hidden_size = hidden_size

        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.candidate_hidden = nn.Linear(input_size + hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, input_tensor: torch.Tensor, h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        outputs = []
        for i in range(input_tensor.size(0)):
            x_t = input_tensor[i].unsqueeze(0)
            
            combined = torch.cat((h_prev, x_t), dim=2).squeeze(0)
            
            r_t = torch.sigmoid(self.reset_gate(combined))
            
            z_t = torch.sigmoid(self.update_gate(combined))
            
            combined_reset = torch.cat((r_t * h_prev.squeeze(0), x_t.squeeze(0)), dim=1)
            n_t = torch.tanh(self.candidate_hidden(combined_reset))
            
            h_t = (1 - z_t) * n_t + z_t * h_prev.squeeze(0)

            outputs.append(h_t)

            h_prev = h_t.unsqueeze(0)
            
        output_sequence = torch.stack(outputs, dim=0)

        output_sequence = self.log_softmax(self.out(output_sequence))

        return output_sequence, h_prev

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size) 