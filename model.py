import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    """
    This is the simple LSTM architecture
    """

    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        padding_idx,
        batch_first,
    ):
        """
        Initialize the model by settingg up the various layers.
        """
        super().__init__()
        self.batch_first = batch_first
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=self.batch_first,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def forward(self, tweet, length):
        """
        Perform a forward pass of our model on input.
        """
        embeds = self.dropout(self.embedding(tweet))
        # embeds = [batch_size, length, embed dim]
        packed_embeds = pack_padded_sequence(
            embeds, length, batch_first=self.batch_first, enforce_sorted=False
        )

        lstm_out, (hidden, _) = self.lstm(packed_embeds)
        output, output_lengths = pad_packed_sequence(
            lstm_out, batch_first=self.batch_first
        )
        # output = [batch_size, length, hidden_dim * num]

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        result = self.fc(hidden)

        # output size [batch_size,  length, hidden_dim]
        return self.sig(result)
