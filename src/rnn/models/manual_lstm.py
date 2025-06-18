        hidden_tuple: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the Manual LSTM model.

        Args:
            input_tensor: Input for a single time step, shape (1, input_size).
            hidden_tuple: Tuple containing (hidden_state, cell_state) from the
                          previous time step, each of shape (1, hidden_size).

        Returns:
            A tuple containing the output tensor and the new (hidden, cell) state tuple.
        """
        hidden_state, cell_state = hidden_tuple
        combined = torch.cat((input_tensor, hidden_state), 1)

        forget_gate_out = torch.sigmoid(self.forget_gate(combined))
        input_gate_out = torch.sigmoid(self.input_gate(combined))
        cell_gate_out = torch.tanh(self.cell_gate(combined))
        output_gate_out = torch.sigmoid(self.output_gate(combined))

        cell_state = forget_gate_out * cell_state + input_gate_out * cell_gate_out
        hidden_state = output_gate_out * torch.tanh(cell_state)

        return hidden_state, (hidden_state, cell_state) 