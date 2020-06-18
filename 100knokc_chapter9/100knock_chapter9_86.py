import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')

# CNN を作成し、保持するクラスです
class MyCNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        # Embedding で単語を埋め込みベクトルに変換します
        self.embed = nn.Embedding(input_size, hidden_size1)
        # CNN の後に活性化関数として tanh をかけます
        self.cnn = nn.Conv1d(in_channels=1, out_channels=hidden_size2, kernel_size=3 * hidden_size1, stride=hidden_size1, padding=hidden_size1)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size2, output_size)

    def forward(self, input):
        # 単語を埋め込みベクトルに変換します
        h1 = self.embed(input)
        # 埋め込みベクトルを CNN の入力の形に変えます
        # 求めた単語ベクトルを全部繋げて、CNN でトークンサイズ(hidden_size1)でスライドすることで単語単位での CNN を実現してます
        h1 = torch.cat([h for h in h1], dim=0)
        h1 = torch.unsqueeze(h1, 0)
        h1 = torch.unsqueeze(h1, 0)
        # CNN
        h2 = self.cnn(h1)
        h2 = self.tanh(h2)
        # maxpool で一つだけを出力する方法が分からなかったので、max で求めました
        h2 = torch.max(h2[0], 1)[0]
        # Linear
        output = self.out(h2)
        return output

def main():
    # 単語のＩＤ化に 80 のプログラムを使い、対応表として辞書 word_id_dict を得ます
    word_id_dict = chapter9_80.get_word_id_dict('../100knock_chapter6/train.feature.txt')
    vocab = max(word_id_dict.values()) + 1

    # CNN を作ります
    INPUT_SIZE = vocab
    HIDDEN_SIZE1 = 300
    HIDDEN_SIZE2 = 50
    OUTPUT_SIZE = 4
    model = MyCNN(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)

    # 例として訓練データの最後の title を入力文として出力結果を得ます
    words = ['France', "'", 's', 'Godard', 'at', '83', 'screens', 'a', '3', 'D', "'", 'Adieu', "'", 'at', 'Cannes']
    words_vector = torch.LongTensor([word_id_dict.get(word, 0) for word in words])
    output = model.forward(words_vector)
    output = torch.nn.functional.softmax(output, dim=0)
    print(output)

    # 出力例
    # tensor([0.3081, 0.1675, 0.3117, 0.2127], grad_fn=<SoftmaxBackward>)

if __name__ == '__main__':
    main()