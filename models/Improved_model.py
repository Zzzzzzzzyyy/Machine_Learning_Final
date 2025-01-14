import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_length, num_layers, nhead=4):
        super(ImprovedModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size * 2, output_length)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        attn_output, attn_output_weights = self.multihead_attention(lstm_out, lstm_out, lstm_out)
        context_vector = self.layer_norm(attn_output)
        context_vector = self.dropout(context_vector)
        output = self.fc(context_vector)
        output = output[:, -1, :]
        output = self.sigmoid(output)

        return output
