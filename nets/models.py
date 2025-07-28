import torch
import torch.nn as nn
# import torch.nn.functional as func
# from collections import OrderedDict

# Define lstm NETWORK structure
class lstm_uni_attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pre_length, seq_length):
        super(lstm_uni_attention, self).__init__()

        self.atten = nn.Sequential(nn.Tanh(), nn.Linear(input_size, input_size), nn.Softmax(2))

        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), int(hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 4), pre_length)
        )

    def forward(self, x):
        wei = self.atten(x).mean(0).mean(0)  # 求注意力矩阵
        x = torch.mul(x, wei)  # 注意力矩阵与特征矩阵相乘
        r_out, hn = self.rnn1(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.fc(r_out[:, -1])
        return out