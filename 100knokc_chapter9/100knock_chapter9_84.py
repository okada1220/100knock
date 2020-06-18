import gensim
import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')

# 単語ベクトルを導入するための make_dataset です
# words_vector_list には各単語の ID が入ってます
# category_vector_list にはカテゴリの対応する数字(0~3)が入ってます
def make_dataset2(filename, word_id_dict, category_table):
    words_vector_list = []
    category_vector_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            words = title.strip().split()
            words_vector = torch.stack([torch.LongTensor([word_id_dict.get(word, 0)]) for word in words], dim=0)
            category_vector = torch.Tensor([category_table[category]])
            words_vector_list.append(words_vector)
            category_vector_list.append(category_vector)
    return words_vector_list, category_vector_list

# RNN を作成し、保持するクラスです
class MyRNN4(nn.Module):
    def __init__(self, word_vectorizer, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        weights = word_vectorizer.wv.syn0
        # Embeddnig で単語 ID から埋め込みベクトルに変換します
        # 事前学習済みの単語ベクトル word_vectorizer で初期化しています
        self.embed = nn.Embedding(weights.shape[0], weights.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(weights))
        # RNN の活性化関数には tanh を使います
        self.rnn = nn.RNN(weights.shape[1], hidden_size, num_layers=1, nonlinearity='tanh')
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = self.embed(input)
        output, self.hidden = self.rnn(input, self.hidden)
        output = self.out(output)
        return output

    # 隠れ状態を初期化します
    def reset(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size)

def main():
    # 単語のＩＤ化に 80 のプログラムを使い、対応表として辞書 word_id_dict を得ます
    word_id_dict = chapter9_80.get_word_id_dict('../100knock_chapter6/train.feature.txt')

    # 事前学習済みの単語ベクトルを読み込みます
    word_vectorizer = gensim.models.KeyedVectors.load_word2vec_format(
        '../100knock_chapter7/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)

    # category を対応する数字にします
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    # RNN を作って、隠れ状態ベクトルを初期化します
    HIDDEN_SIZE = 50
    OUTPUT_SIZE = 4
    model = MyRNN4(word_vectorizer, HIDDEN_SIZE, OUTPUT_SIZE)
    model.reset()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 学習データを読み込みます
    words_vector_list, category_vector_list = make_dataset2('../100knock_chapter6/mini-train.feature.txt', word_id_dict, category_table)

    # エポック回数３回で学習させます
    EPOCH_NUM = 3
    for epoch in range(EPOCH_NUM):
        accuracy_sum = 0
        loss_sum = 0
        for i in range(len(words_vector_list)):
            # 初期化します
            model.reset()
            optimizer.zero_grad()
            # 各事例で損失を求め、重みを更新します
            words_vector = words_vector_list[i]
            category_vector = category_vector_list[i]
            pred_category_vector = model.forward(words_vector)
            loss = criterion(pred_category_vector[-1], category_vector.type(torch.long))
            loss.backward()
            optimizer.step()
            loss_sum += loss
            accuracy_sum += int(category_vector_list[i] == torch.argmax(pred_category_vector[-1][-1]))
        # 正答率と損失の平均を求めます
        accuracy = accuracy_sum / len(words_vector_list)
        print('accuracy(train):', accuracy)
        loss_average = loss_sum / len(words_vector_list)
        print('loss(train):', loss_average)

    # 評価データを読み込みます
    words_vector_list, category_vector_list = make_dataset2('../100knock_chapter6/mini-test.feature.txt', word_id_dict, category_table)

    # 初期化します
    accuracy_sum = 0
    loss_sum = 0
    model.reset()
    for i in range(len(words_vector_list)):
        words_vector = words_vector_list[i]
        category_vector = category_vector_list[i]
        pred_category_vector = model.forward(words_vector)
        loss = criterion(pred_category_vector[-1], category_vector.type(torch.long))
        loss_sum += loss
        accuracy_sum += int(category_vector_list[i] == torch.argmax(pred_category_vector[-1][-1]))
    # 正答率と損失の平均を求めます
    accuracy = accuracy_sum / len(words_vector_list)
    print('accuracy(test):', accuracy)
    loss_average = loss_sum / len(words_vector_list)
    print('loss(test):', loss_average)

if __name__ == '__main__':
    main()