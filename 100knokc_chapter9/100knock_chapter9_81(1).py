# from torch.nn.functional import one_hot
import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')

# RNN を作成し、保持するクラスです
class MyRNN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # RNN の活性化関数には tanh を使います
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh')
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, self.hidden = self.rnn(input, self.hidden)
        output = self.out(output)
        return output

    # 隠れ状態を初期化します
    def reset(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size)

# 単語を word_id_idct を使ってＩＤ化し、埋め込み行列 word_matrix を使って get_vector により単語の埋め込みベクトルを返します
class WordVectorizer:
    def __init__(self, word_id_dict, word_matrix):
        self.word_id_dict = word_id_dict
        self.vocab = max(word_id_dict.values()) + 1
        self.word_matrix = word_matrix

    # 単語 word -> one_hotベクトル word_id -> 埋め込みベクトル word_vector にします
    def get_vector(self, word):
        word_id = self.word_id_dict.get(word, 0)
        """
         torch.nn.functional.one_hot を使うとエラーが発生してしまって困り果てました(型が違うらしいです）
        tensor_id = torch.tensor(word_id)
        one_hot_vector = one_hot(tensor_id, num_classes=self.vocab)
         RuntimeError: Expected object of scalar type Float but got scalar type Long for argument #3 'vec' in call to _th_addmv_out
        """
        one_hot_vector = torch.eye(self.vocab)[word_id]
        word_vector = torch.mv(self.word_matrix, one_hot_vector)
        return word_vector

def main():
    # 単語のＩＤ化に 80 のプログラムを使い、対応表として辞書 word_id_dict を得ます
    word_id_dict = chapter9_80.get_word_id_dict('../100knock_chapter6/train.feature.txt')

    # ランダムに埋め込み行列を決定します（本来はここに適切な埋め込み行列を用います）
    torch.manual_seed(3)
    word_matrix = torch.randn(300, max(word_id_dict.values()) + 1)

    # 単語を埋め込みベクトルにしてくれる word_vectorizer を作ります
    word_vectorizer = WordVectorizer(word_id_dict, word_matrix)

    # RNN を作って、隠れ状態ベクトルを初期化します
    INPUT_SIZE = 300
    HIDDEN_SIZE = 50
    OUTPUT_SIZE = 4
    model = MyRNN1(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model.reset()

    # 例として訓練データの最後の title を入力文として出力結果を得ます
    words = ['France', "'", 's', 'Godard', 'at', '83', 'screens', 'a', '3', 'D', "'", 'Adieu', "'", 'at', 'Cannes']
    words_vector = torch.stack([torch.unsqueeze(word_vectorizer.get_vector(word), 0) for word in words], dim=0)
    print(words_vector)
    output = model.forward(words_vector)
    output = torch.nn.functional.softmax(output, dim=2)
    print(output[-1])

    # tensor([[[-1.3212,  0.7239,  0.5369,  ...,  1.4811,  1.1291, -1.4901]],
    #
    #         [[-0.0766,  0.1435, -0.6699,  ...,  1.1296, -0.1385, -0.3836]],
    #
    #         [[ 0.3599, -0.4606, -0.7770,  ...,  1.4223, -0.3953,  0.7650]],
    #
    #         ...,
    #
    #         [[-0.0766,  0.1435, -0.6699,  ...,  1.1296, -0.1385, -0.3836]],
    #
    #         [[ 0.5222,  1.0859,  0.3182,  ..., -0.3919,  0.0468, -0.3743]],
    #
    #         [[-0.7885,  0.1760,  0.0610,  ...,  0.5607,  1.0050, -0.7504]]])

    # tensor([[0.1848, 0.3809, 0.1387, 0.2955]], grad_fn=<SelectBackward>)

if __name__ == '__main__':
    main()