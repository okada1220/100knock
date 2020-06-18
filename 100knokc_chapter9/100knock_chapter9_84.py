import gensim
import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')
chapter9_83 = __import__('100knock_chapter9_83')

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
        print(input)
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
    words_vector_list, category_vector_list = chapter9_83.make_dataset_embed('../100knock_chapter6/minimini-train.feature.txt', word_id_dict, category_table)

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

    # tensor([[ 815],
    #         [ 816],
    #         [   7],
    #         [  92],
    #         [5212],
    #         [   0],
    #         [  26],
    #         [  23],
    #         [3047],
    #         [ 673],
    #         [  29],
    #         [   0]])
    # tensor([[[ 8.3496e-02, -7.2937e-03, -2.6489e-02,  ...,  3.1494e-02,
    #           -2.6489e-02,  5.1270e-02]],
    #
    #         [[ 5.6885e-02,  1.0742e-01, -6.4941e-02,  ..., -2.7222e-02,
    #            1.0681e-02, -2.3145e-01]],
    #
    #         [[-1.7285e-01,  2.7930e-01,  1.0693e-01,  ...,  1.2305e-01,
    #            1.2988e-01, -1.8262e-01]],
    #
    #         ...,
    #
    #         [[ 7.0801e-02, -3.4912e-02,  6.5430e-02,  ..., -2.0801e-01,
    #           -9.3262e-02, -1.7188e-01]],
    #
    #         [[ 2.3438e-02, -1.4062e-01,  1.6113e-01,  ..., -1.5625e-01,
    #           -2.0905e-03,  7.9590e-02]],
    #
    #         [[ 1.1292e-03, -8.9645e-04,  3.1853e-04,  ..., -1.5640e-03,
    #           -1.2302e-04, -8.6308e-05]]], grad_fn=<EmbeddingBackward>)

if __name__ == '__main__':
    main()