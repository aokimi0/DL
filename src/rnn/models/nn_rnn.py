import torch
import torch.nn as nn
from typing import Tuple


class NNRNN(nn.Module):
    """
    A wrapper around the standard PyTorch nn.RNN module to ensure a consistent
    interface with the training script.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(NNRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self, input_tensor: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the RNN model.

        Args:
            input_tensor: Input tensor for a single time step,
                          shape (batch_size, seq_len, input_size).
            hidden: Hidden state from the previous time step,
                    shape (1, batch_size, hidden_size).

        Returns:
            A tuple containing the output tensor and the new hidden state.
        """
        output, hidden = self.rnn(input_tensor, hidden)
        # output: (batch_size, seq_len, hidden_size)
        # We only need the output of the last time step
        last_output = output[:, -1, :]
        output = self.fc(last_output)
        output = self.log_softmax(output)
        return output, hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initializes the hidden state."""
        device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.rnn.hidden_size).to(device) 