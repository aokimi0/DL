        hidden_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Manual GRU model.

        Args:
            input_tensor: Input tensor for a single time step.
                          Shape should be (batch_size, input_size), e.g., (1, 57).
            hidden_tensor: Hidden state from the previous time step.
                           Shape (batch_size, hidden_size), e.g., (1, 128).

        Returns:
            A tuple containing the output tensor and the new hidden state.
        """
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        reset_gate_out = torch.sigmoid(self.reset_gate(combined))
        update_gate_out = torch.sigmoid(self.update_gate(combined)) 