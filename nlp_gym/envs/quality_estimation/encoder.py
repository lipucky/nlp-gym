import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Base Encoder Class"""


class GRUEncoder(Encoder):
    def __init__(self, vocab_size: int, hidden_size: int, device: str = "cpu"):
        self.device = device

        super(GRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = vocab_size
        self.embedding = nn.Embedding(self.input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
