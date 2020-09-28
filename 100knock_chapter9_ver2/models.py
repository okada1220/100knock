# -*- coding: utf-8 -*-

import torch
from torch import nn

# 初期のモデルのパラメータがプログラムごとで変化しないように固定します
torch.manual_seed(0)

# 通常の RNNモデルです
class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size,
                          num_layers=1, nonlinearity='tanh', batch_first=True)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        embed_input = self.embed(input)
        output, self.hidden = self.rnn(embed_input, self.hidden)
        output = self.linear(output[:, -1, :])
        output = self.softmax(output)
        return output

    def reset(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size)

# 最後の softmax を除いた RNNモデルです（損失関数のクロスエントロピが softmax を含んでいるため）
class RNN_simple(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size,
                          num_layers=1, nonlinearity='tanh', batch_first=True)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embed_input = self.embed(input)
        output, self.hidden = self.rnn(embed_input, self.hidden)
        output = self.linear(output[:, -1, :])
        return output

    def reset(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size)

# 100knock_chapter9_ver2_84 で使う RNNモデルです
class RNN_simple_minibatch_2(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size,
                          num_layers=1, nonlinearity='tanh', batch_first=True)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, index):
        embed_input = self.embed(input)
        padding_input = nn.utils.rnn.pack_padded_sequence(embed_input, index, batch_first=True)
        output, self.hidden = self.rnn(padding_input, self.hidden)
        output = self.linear(self.hidden[-1])
        return output

    def reset(self, batch_size):
        self.hidden = torch.zeros(1, batch_size, self.hidden_size)

# 100knock_chapter9_ver2_83(1) で使う RNNモデルです
class RNN_simple_minibatch_gpu(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size, padding_idx=padding_idx)
        self.rnn = nn.RNN(embed_size, hidden_size,
                          num_layers=1, nonlinearity='tanh', batch_first=True)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, index):
        embed_input = self.embed(input)
        output, hidden = self.rnn(embed_input, self.hidden)
        output = self.linear(output[list(range(len(input))), index-1, :])
        return output

    def reset(self, batch_size):
        self.hidden = torch.zeros(1, batch_size, self.hidden_size).cuda()

# 100knock_chapter9_ver2_83(2) で使う RNNモデルです
class RNN_simple_minibatch_gpu_2(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size,
                          num_layers=1, nonlinearity='tanh', batch_first=True)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, index):
        embed_input = self.embed(input)
        padding_input = nn.utils.rnn.pack_padded_sequence(embed_input, index, batch_first=True)
        output, self.hidden = self.rnn(padding_input, self.hidden)
        output = self.linear(self.hidden[-1])
        return output

    def reset(self, batch_size):
        self.hidden = torch.zeros(1, batch_size, self.hidden_size).cuda()

# 100knock_chapter9_ver2_84 で使う多層双方向 RNN モデルです
class MultiRNN_simple_minibatch(nn.Module):
    def __init__(self, num_layers, input_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size,
                          num_layers=num_layers, nonlinearity='tanh', batch_first=True, bidirectional=True)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size * num_layers * 2, output_size)

    def forward(self, input, index):
        embed_input = self.embed(input)
        padding_input = nn.utils.rnn.pack_padded_sequence(embed_input, index, batch_first=True)
        _, self.hidden = self.rnn(padding_input, self.hidden)
        cat_hidden = torch.cat([self.hidden[i].squeeze(0) for i in range(self.num_layers * 2)]).unsqueeze(0)
        output = self.linear(cat_hidden)
        return output

    def reset(self, batch_size):
        self.hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)

# 通常の CNNモデルです
class CNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.cnn = nn.Conv2d(1, hidden_size, kernel_size=(3, embed_size), stride=1, padding=(1, 0), bias=True)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input):
        embed_input = self.embed(input)
        embed_input = embed_input.unsqueeze(0)  #(channel, height, width) -> (batch, channel, height, width)
        cnn_output = self.cnn(embed_input)
        tanh_output = self.tanh(cnn_output)
        tanh_output = tanh_output.squeeze(0)  #(batch, channel, height, 1) -> (channel, height, 1)
        maxpool_output = nn.functional.max_pool2d(tanh_output, (len(input.squeeze(0)), 1))
        maxpool_output = maxpool_output.squeeze(-1).squeeze(-1)  #(channel, height) -> (channel, 1)  -> (channel)
        linear_output = self.linear(maxpool_output)
        output = self.softmax(linear_output)
        return output

# 最後の softmax を除いた RNNモデルです（損失関数のクロスエントロピが softmax を含んでいるため）
class CNN_simple(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.cnn = nn.Conv2d(1, hidden_size, kernel_size=(3, embed_size), stride=1, padding=(1, 0), bias=True)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embed_input = self.embed(input)
        embed_input = embed_input.unsqueeze(0)  #(channel, height, width) -> (batch, channel, height, width)
        cnn_output = self.cnn(embed_input)
        tanh_output = self.tanh(cnn_output)
        tanh_output = tanh_output.squeeze(0)  #(batch, channel, height, 1) -> (channel, height, 1)
        maxpool_output = nn.functional.max_pool2d(tanh_output, (len(input.squeeze(0)), 1))
        maxpool_output = maxpool_output.squeeze(-1).squeeze(-1)  #(channel, height) -> (channel, 1)  -> (channel)
        linear_output = self.linear(maxpool_output)
        return linear_output

# 100knock_chapter9_ver2_88 で使うニューラルネットワークモデルです
class MultiNet_simple(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size):
        super().__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.cnn = nn.Conv2d(1, hidden_size, kernel_size=(3, embed_size), stride=1, padding=(1, 0), bias=True)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embed_input = self.embed(input)
        embed_input = embed_input.unsqueeze(0)  # (channel, height, width) -> (batch, channel, height, width)
        cnn_output = self.cnn(embed_input)
        tanh_output = self.tanh(cnn_output)
        tanh_output = tanh_output.squeeze(0)  # (batch, channel, height, 1) -> (channel, height, 1)
        maxpool_output = nn.functional.max_pool2d(tanh_output, (len(input.squeeze(0)), 1))
        maxpool_output = maxpool_output.squeeze(-1).squeeze(-1)  # (channel, height) -> (channel, 1)  -> (channel)
        linear_output = self.linear(maxpool_output)
        return linear_output