import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')

# RNN を作成し、保持するクラスです
class MyRNN2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        # Linear を使って単語の one-hot ベクトルから埋め込みベクトルを得ます
        # -> Embedding を使った方がいい？（入力の形を変更する必要あり）（保留中）
        self.input = nn.Linear(input_size, hidden_size1)
        # RNN の活性化関数には tanh を使います
        self.rnn = nn.RNN(hidden_size1, hidden_size2, num_layers=1, nonlinearity='tanh')
        self.out = nn.Linear(hidden_size2, output_size)

    def forward(self, input):
        input = self.input(input)
        output, self.hidden = self.rnn(input, self.hidden)
        output = self.out(output)
        return output

    # 隠れ状態を初期化します
    def reset(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size2)

def main():
    # 単語のＩＤ化に 80 のプログラムを使い、対応表として辞書 word_id_dict を得ます
    word_id_dict = chapter9_80.get_word_id_dict('../100knock_chapter6/train.feature.txt')
    vocab = max(word_id_dict.values()) + 1

    # RNN を作って、隠れ状態ベクトルを初期化します
    INPUT_SIZE = vocab
    HIDDEN_SIZE1 = 300
    HIDDEN_SIZE2 = 50
    OUTPUT_SIZE = 4
    model = MyRNN2(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
    model.reset()

    # 例として訓練データの最後の title を入力文として出力結果を得ます
    words = ['France', "'", 's', 'Godard', 'at', '83', 'screens', 'a', '3', 'D', "'", 'Adieu', "'", 'at', 'Cannes']
    words_vector = torch.stack([torch.unsqueeze(torch.eye(vocab)[word_id_dict.get(word, 0)], 0) for word in words], dim=0)
    print(words_vector)
    output = model.forward(words_vector)
    output = torch.nn.functional.softmax(output, dim=2)
    print(output[-1])

    # tensor([[[0., 0., 0.,  ..., 0., 0., 0.]],
    #
    #         [[1., 0., 0.,  ..., 0., 0., 0.]],
    #
    #         [[0., 1., 0.,  ..., 0., 0., 0.]],
    #
    #         ...,
    #
    #         [[1., 0., 0.,  ..., 0., 0., 0.]],
    #
    #         [[0., 0., 0.,  ..., 0., 0., 0.]],
    #
    #         [[0., 0., 0.,  ..., 0., 0., 0.]]])

    # tensor([[0.2192, 0.2717, 0.2312, 0.2779]], grad_fn=<SelectBackward>)

if __name__ == '__main__':
    main()