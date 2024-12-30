"""LSTM from PyTorch is not compatible with vmap
(gives error: RuntimeError: accessing `data` under vmap transform is not allowed)
Instead use custom LSTM cell from https://github.com/normal-computing/posteriors/blob/main/examples/imdb/lstm.py
"""
import torch
import torch.nn as nn

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, states):
        h_prev, c_prev = states
        combined = torch.cat((input, h_prev), 1)

        f_t = torch.sigmoid(self.forget_gate(combined))
        i_t = torch.sigmoid(self.input_gate(combined))
        c_tilde = torch.tanh(self.cell_gate(combined))
        c_t = f_t * c_prev + i_t * c_tilde
        o_t = torch.sigmoid(self.output_gate(combined))
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.lstm_cell = CustomLSTMCell(input_size, hidden_size)

    def forward(self, input, initial_states=None):
        if initial_states is None:
            initial_h = torch.zeros(
                1,
                input.size(0 if self.batch_first else 1),
                self.hidden_size,
                device=input.device,
            )
            initial_c = torch.zeros(
                1,
                input.size(0 if self.batch_first else 1),
                self.hidden_size,
                device=input.device,
            )
        else:
            initial_h, initial_c = initial_states

        # Ensure we are working with single layer, single direction states
        initial_h = initial_h.squeeze(0)
        initial_c = initial_c.squeeze(0)

        if self.batch_first:
            input = input.transpose(
                0, 1
            )  # Convert (batch, seq_len, feature) to (seq_len, batch, feature)

        outputs = []
        h_t, c_t = initial_h, initial_c

        for i in range(
            input.shape[0]
        ):  # input is expected to be (seq_len, batch, input_size)
            h_t, c_t = self.lstm_cell(input[i], (h_t, c_t))
            outputs.append(h_t.unsqueeze(0))

        outputs = torch.cat(outputs, 0)

        if self.batch_first:
            outputs = outputs.transpose(
                0, 1
            )  # Convert back to (batch, seq_len, feature)

        return outputs, (h_t, c_t)
