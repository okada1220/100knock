from torch import nn

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