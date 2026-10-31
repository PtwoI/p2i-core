"""Offline recurrent classifiers with an editable recurrent module and head."""
from torch import nn


class RecurrentClassifier(nn.Module):
    def __init__(self, kind='lstm'):
        super().__init__()
        self.kind = kind
        cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[kind]
        self.recurrent = cls(8, 16, batch_first=True)
        self.head = nn.Linear(16, 3)

    def forward(self, x):
        output, _ = self.recurrent(x)
        return self.head(output[:, -1])
