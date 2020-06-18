import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')

# BiRNN を作成し、保持するクラスです
class MyBiRNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        # Linear を使って単語の one-hot ベクトルから埋め込みベクトルを得ます
        self.input = nn.Linear(input_size, hidden_size1)
        # BiRNN の活性化関数には tanh を使います
        self.rnn = nn.RNN(hidden_size1, hidden_size2, num_layers=1, nonlinearity='tanh', bidirectional=True)
        self.out = nn.Linear(hidden_size2 * 2, output_size)

    def forward(self, input):
        input = self.input(input)
        _, self.hidden = self.rnn(input, self.hidden)
        output = torch.cat([self.hidden[0][0], self.hidden[1][0]])
        output = self.out(output)
        return output

    # 隠れ状態を初期化します
    def reset(self):
        self.hidden = torch.zeros(2, 1, self.hidden_size2)

def main():
    # 単語のＩＤ化に 80 のプログラムを使い、対応表として辞書 word_id_dict を得ます
    word_id_dict = chapter9_80.get_word_id_dict('../100knock_chapter6/train.feature.txt')
    vocab = max(word_id_dict.values()) + 1

    # RNN を作って、隠れ状態ベクトルを初期化します
    INPUT_SIZE = vocab
    HIDDEN_SIZE1 = 300
    HIDDEN_SIZE2 = 50
    OUTPUT_SIZE = 4
    model = MyBiRNN(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
    model.reset()

    # 例として訓練データの最後の title を入力文として出力結果を得ます
    words = ['France', "'", 's', 'Godard', 'at', '83', 'screens', 'a', '3', 'D', "'", 'Adieu', "'", 'at', 'Cannes']
    words_vector = torch.stack([torch.unsqueeze(torch.eye(vocab)[word_id_dict.get(word, 0)], 0) for word in words], dim=0)
    output = model.forward(words_vector)
    output = torch.nn.functional.softmax(output, dim=0)
    print(output)

    # 出力例
    # tensor([0.2375, 0.2682, 0.2741, 0.2202], grad_fn=<SoftmaxBackward>)

if __name__ == '__main__':
    main()