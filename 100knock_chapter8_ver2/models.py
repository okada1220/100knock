import torch
from torch import nn

# 初期のモデルのパラメータがプログラムごとで変化しないように固定します
torch.manual_seed(0)

# 単層ニューラルネットワーク
class SingleNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, output_size, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.input(input)
        output = self.softmax(output)
        return output

# 単層ニューラルネットワーク（ Softmax 除去）
class SingleNet_simple(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, output_size, bias=False)

    def forward(self, input):
        output = self.input(input)
        return output

# 多層ニューラルネットワーク（ Softmax 除去）
class Net_simple(nn.Module):
    def __init__(self, input_size, output_size, layer_num):
        super().__init__()
        modulelist = []
        for _ in range(layer_num):
            modulelist.append(nn.Linear(input_size, input_size, bias=False))
            modulelist.append(nn.ReLU())
        self.layers = nn.ModuleList(modulelist)
        self.output = nn.Linear(input_size, output_size)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        output = self.output(input)
        return output
