import gensim
import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')
chapter9_83 = __import__('100knock_chapter9_83')
import pprint

# RNN を作成し、保持するクラスです
class MyRNN4(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.hidden_size2 = hidden_size2
        # Embeddnig で単語 ID から埋め込みベクトルに変換します
        self.embed = nn.Embedding(input_size, hidden_size1)
        # RNN の活性化関数には tanh を使います
        self.rnn = nn.RNN(hidden_size1, hidden_size2, num_layers=1, nonlinearity='tanh')
        self.out = nn.Linear(hidden_size2, output_size)

    def forward(self, input):
        input = self.embed(input)
        output, self.hidden = self.rnn(input, self.hidden)
        output = self.out(output)
        return output

    # 隠れ状態を初期化します
    def reset(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size2)

    def embedset(self, word_matrix):
        self.embed.weight = nn.Parameter(word_matrix)

def main():
    # 単語のＩＤ化に 80 のプログラムを使い、対応表として辞書 word_id_dict を得ます
    word_id_dict = chapter9_80.get_word_id_dict('../100knock_chapter6/train.feature.txt')
    vocab = max(word_id_dict.values()) + 1

    # 事前学習済みの単語ベクトルを読み込みます
    word_vectorizer = gensim.models.KeyedVectors.load_word2vec_format(
        '../100knock_chapter7/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
    weights = word_vectorizer.wv.syn0

    word_matrix = torch.zeros([vocab, weights.shape[1]])
    pprint.pprint(word_matrix)
    for word, id in word_id_dict.items():
        if id == 0:
            continue
        if word in word_vectorizer:
            word_matrix[id] = torch.Tensor(word_vectorizer[word])
    pprint.pprint(word_matrix)

    # category を対応する数字にします
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    # RNN を作って、隠れ状態ベクトルを初期化します
    INPUT_SIZE = vocab
    HIDDEN_SIZE1 = weights.shape[1]
    HIDDEN_SIZE2 = 50
    OUTPUT_SIZE = 4
    model = MyRNN4(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
    model.reset()
    model.embedset(word_matrix)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 学習データを読み込みます
    words_vector_list, category_vector_list = chapter9_83.make_dataset_embed('../100knock_chapter6/mini-train.feature.txt', word_id_dict, category_table)

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
    words_vector_list, category_vector_list = chapter9_83.make_dataset_embed('../100knock_chapter6/mini-test.feature.txt', word_id_dict, category_table)

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

    # mini-train.feature.txtでの学習結果（1000文）
    #   accuracy(train): 0.722
    #   loss(train): tensor(0.8122, grad_fn=<DivBackward0>)
    #   accuracy(train): 0.87
    #   loss(train): tensor(0.3812, grad_fn=<DivBackward0>)
    #   accuracy(train): 0.971
    #   loss(train): tensor(0.1108, grad_fn=<DivBackward0>)

    # mini-test.feature.txtでのテスト結果（500文）
    #   accuracy(test): 0.766
    #   loss(test): tensor(0.7935, grad_fn=<DivBackward0>)

if __name__ == '__main__':
    main()