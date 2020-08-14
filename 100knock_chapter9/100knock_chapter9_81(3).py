# Embedding と TensorDataset,Dateloader を理解したいプログラム
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
chapter9_80 = __import__('100knock_chapter9_80')

# RNN を作成し、保持するクラスです
class MyRNN3(nn.Module):
    def __init__(self, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        # self.input = nn.Embedding(input_size, hidden_size1)
        # RNN の活性化関数には tanh を使います
        self.rnn = nn.RNN(hidden_size1, hidden_size2, num_layers=1, nonlinearity='tanh', batch_first=True)
        self.out = nn.Linear(hidden_size2, output_size)

    def forward(self, input):
        #input = self.input(input)
        #print(input)
        output, self.hidden = self.rnn(input, self.hidden)
        output = self.out(output)
        return output

    # 隠れ状態を初期化します
    def reset(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size2)

def get_dataset(filename, word_id_dict, category_table, embed):
    words_vector_list = []
    category_vector_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            words = title.strip().split()
            words = [word_id_dict.get(word, 0) for word in words]
            words_vector = torch.LongTensor(words)
            words_vector = embed(words_vector)
            category_vector = torch.Tensor([category_table[category]])
            words_vector_list.append(words_vector)
            category_vector_list.append(category_vector)
        category_vector_list = torch.stack(category_vector_list, dim=0)
        words_vector_list = torch.stack(words_vector_list, dim=0)
        # minibatch は同じバッチサイズじゃないとダメ
        # -> Ｂ事例でまとめようとすると各事例で語数が違うから無理
        # 各事例ごとにニューラルネットワークを使ってカテゴリを予測する
        # -> 事例ごとで別のニューラルネットワークを利用しているようなもの
        # -> まとめて一つのニューラルネットワークにまかせること自体が無理？
        dataset = TensorDataset(words_vector_list, category_vector_list)
    return dataset

def main():
    # 単語のＩＤ化に 80 のプログラムを使い、対応表として辞書 word_id_dict を得ます
    word_id_dict = chapter9_80.get_word_id_dict('../100knock_chapter6/train.feature.txt')
    vocab = max(word_id_dict.values()) + 1

    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    embed = nn.Embedding(vocab, 300)

    # RNN を作って、隠れ状態ベクトルを初期化します
    #INPUT_SIZE = vocab
    HIDDEN_SIZE1 = 300
    HIDDEN_SIZE2 = 50
    OUTPUT_SIZE = 4
    model = MyRNN3(HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
    model.reset()

    # 例として訓練データの最後の title を入力文として出力結果を得ます
    words = ['France', "'", 's', 'Godard', 'at', '83', 'screens', 'a', '3', 'D', "'", 'Adieu', "'", 'at', 'Cannes']
    category = 'e'
    words = [word_id_dict.get(word, 0) for word in words]
    print(words)
    category = category_table[category]
    words_vector = torch.stack([torch.LongTensor(words)], dim=0)
    words_vector = embed(words_vector)
    category_vector = torch.unsqueeze(torch.Tensor([category]), 0)
    print(words_vector)
    print(category_vector)
    dataset = TensorDataset(words_vector, category_vector)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dataloader):
        print(i)
        print(data)
        w, c = data
        output = model.forward(w)
    output = torch.nn.functional.softmax(output, dim=2)
    print(output)
    print(output[0][-1])

if __name__ == '__main__':
    main()