import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')
chapter9_81_2 = __import__('100knock_chapter9_81(2)')

# file から one-hot ベクトルにしてモデルへの入力の形に合わせて list に入れました
def make_dataset(filename, word_id_dict, category_table):
    words_vector_list = []
    category_vector_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            words = title.strip().split()
            words_vector = torch.stack([torch.unsqueeze(torch.eye(max(word_id_dict.values())+1)[word_id_dict.get(word, 0)], 0) for word in words], dim=0)
            category_vector = torch.Tensor([category_table[category]])
            words_vector_list.append(words_vector)
            category_vector_list.append(category_vector)
    return words_vector_list, category_vector_list

def main():
    # 単語のＩＤ化に 80 のプログラムを使い、対応表として辞書 word_id_dict を得ます
    word_id_dict = chapter9_80.get_word_id_dict('../100knock_chapter6/train.feature.txt')
    vocab = max(word_id_dict.values()) + 1

    # category を対応する数字にします
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    # RNN を作って、隠れ状態ベクトルを初期化します
    INPUT_SIZE = vocab
    HIDDEN_SIZE1 = 300
    HIDDEN_SIZE2 = 50
    OUTPUT_SIZE = 4
    model = chapter9_81_2.MyRNN2(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 学習データを読み込みます
    words_vector_list, category_vector_list = make_dataset('../100knock_chapter6/train.feature.txt', word_id_dict, category_table)

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
    words_vector_list, category_vector_list = make_dataset('../100knock_chapter6/test.feature.txt', word_id_dict, category_table)

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
    # accuracy(train): 0.529
    # loss(train): tensor(1.1248, grad_fn=<DivBackward0>)
    # accuracy(train): 0.852
    # loss(train): tensor(0.4471, grad_fn=<DivBackward0>)
    # accuracy(train): 0.961
    # loss(train): tensor(0.1110, grad_fn=<DivBackward0>)

    # mini-test.feature.txtでのテスト結果（500文）
    # accuracy(test): 0.542
    # loss(test): tensor(1.5572, grad_fn=<DivBackward0>)

if __name__ == '__main__':
    main()