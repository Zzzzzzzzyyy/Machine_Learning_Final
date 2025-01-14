import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_length, hidden_size, num_layers):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size*2, output_length)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        dropped_out = self.dropout(last_hidden_state)
        output = self.fc(dropped_out)
        output = self.sigmod(output)
        return output

