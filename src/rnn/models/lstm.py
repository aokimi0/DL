import torch
import torch.nn as nn
from typing import Tuple

class ManualLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(ManualLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.gates = nn.Linear(input_size + hidden_size, hidden_size * 4)
        
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, input_tensor: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        h_prev, c_prev = hidden_state
        
        outputs = []
        for i in range(input_tensor.size(0)):
            x_t = input_tensor[i].unsqueeze(0)
            
            combined = torch.cat((h_prev, x_t), dim=2)
            
            all_gates = self.gates(combined.squeeze(0))
            
            f_t = torch.sigmoid(all_gates[:, :self.hidden_size])
            i_t = torch.sigmoid(all_gates[:, self.hidden_size:self.hidden_size*2])
            g_t = torch.tanh(all_gates[:, self.hidden_size*2:self.hidden_size*3])
            o_t = torch.sigmoid(all_gates[:, self.hidden_size*3:])
            
            c_t = f_t * c_prev.squeeze(0) + i_t * g_t
            
            h_t = o_t * torch.tanh(c_t)
            
            outputs.append(h_t)

            h_prev = h_t.unsqueeze(0)
            c_prev = c_t.unsqueeze(0)
            
        output_sequence = torch.stack(outputs, dim=0)

        output_sequence = self.log_softmax(self.out(output_sequence))

        return output_sequence, (h_prev, c_prev)

    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size),
        ) 