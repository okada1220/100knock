import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')

# CNN の入力に合わせたデータを作るための関数です
def make_dataset_cnn(filename, word_id_dict, category_table):
    words_vector_list = []
    category_vector_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            words = title.strip().split()
            # words_vector の形をこれまでと変えてます
            words_vector = torch.LongTensor([word_id_dict.get(word, 0) for word in words])
            category_vector = torch.Tensor([category_table[category]])
            words_vector_list.append(words_vector)
            category_vector_list.append(category_vector)
    return words_vector_list, category_vector_list

# CNN を作成し、保持するクラスです
class MyCNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        # Embedding で単語を埋め込みベクトルに変換します
        self.embed = nn.Embedding(input_size, hidden_size1)
        # CNN の後に活性化関数として tanh をかけます
        # フィルターのサイズを 3 -> 5 に変更しました
        self.cnn = nn.Conv1d(in_channels=1, out_channels=hidden_size2, kernel_size=5 * hidden_size1, stride=hidden_size1, padding=2 * hidden_size1)
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

    # category を対応する数字にします
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    # CNN を作ります
    INPUT_SIZE = vocab
    HIDDEN_SIZE1 = 300
    HIDDEN_SIZE2 = 50
    OUTPUT_SIZE = 4
    model = MyCNN(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 学習データを読み込みます
    words_vector_list, category_vector_list = make_dataset_cnn('../100knock_chapter6/mini-train.feature.txt', word_id_dict, category_table)

    # エポック回数３回で学習させます
    EPOCH_NUM = 3
    for epoch in range(EPOCH_NUM):
        accuracy_sum = 0
        loss_sum = 0
        for i in range(len(words_vector_list)):
            # 初期化します
            optimizer.zero_grad()
            # 各事例で損失を求め、重みを更新します
            words_vector = words_vector_list[i]
            category_vector = category_vector_list[i]
            pred_category = model.forward(words_vector)
            pred_category_vector = torch.unsqueeze(pred_category, 0)
            loss = criterion(pred_category_vector, category_vector.type(torch.long))
            loss.backward()
            optimizer.step()
            loss_sum += loss
            accuracy_sum += int(category_vector_list[i] == torch.argmax(pred_category))
        # 正答率と損失の平均を求めます
        accuracy = accuracy_sum / len(words_vector_list)
        print('accuracy(train):', accuracy)
        loss_average = loss_sum / len(words_vector_list)
        print('loss(train):', loss_average)

    # 評価データを読み込みます
    words_vector_list, category_vector_list = make_dataset_cnn('../100knock_chapter6/mini-test.feature.txt', word_id_dict, category_table)

    # 初期化します
    accuracy_sum = 0
    loss_sum = 0
    for i in range(len(words_vector_list)):
        words_vector = words_vector_list[i]
        category_vector = category_vector_list[i]
        pred_category = model.forward(words_vector)
        pred_category_vector = torch.unsqueeze(pred_category, 0)
        loss = criterion(pred_category_vector, category_vector.type(torch.long))
        loss_sum += loss
        accuracy_sum += int(category_vector_list[i] == torch.argmax(pred_category))
    # 正答率と損失の平均を求めます
    accuracy = accuracy_sum / len(words_vector_list)
    print('accuracy(test):', accuracy)
    loss_average = loss_sum / len(words_vector_list)
    print('loss(test):', loss_average)

    # mini-train.feature.txtでの学習結果（1000文）
    # accuracy(train): 0.529
    # loss(train): tensor(1.1284, grad_fn=<DivBackward0>)
    # accuracy(train): 0.745
    # loss(train): tensor(0.7522, grad_fn=<DivBackward0>)
    # accuracy(train): 0.889
    # loss(train): tensor(0.3397, grad_fn=<DivBackward0>)

    # mini-test.feature.txtでのテスト結果（500文）
    # accuracy(test): 0.736
    # loss(test): tensor(0.8040, grad_fn=<DivBackward0>)

if __name__ == '__main__':
    main()