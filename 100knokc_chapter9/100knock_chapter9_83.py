import torch
from torch import nn
chapter9_80 = __import__('100knock_chapter9_80')
chapter9_81_2 = __import__('100knock_chapter9_81(2)')
chapter9_82 = __import__('100knock_chapter_82')

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
    words_vector_list, category_vector_list = chapter9_82.make_dataset('../100knock_chapter6/train.feature.txt', word_id_dict, category_table)

    # エポック回数３回、ミニバッチ１０で学習させます
    EPOCH_NUM = 3
    BATCH_SIZE = 10
    flag = 0
    for epoch in range(EPOCH_NUM):
        accuracy_sum = 0
        loss_sum = 0
        loss_batch_sum = 0
        for i in range(len(words_vector_list)):
            # 初期化します
            model.reset()
            optimizer.zero_grad()
            # 各事例で損失を求め、loss_batch_sum でバッチサイズ分合計します
            words_vector = words_vector_list[i]
            category_vector = category_vector_list[i]
            pred_category_vector = model.forward(words_vector)
            loss = criterion(pred_category_vector[-1], category_vector.type(torch.long))
            loss_batch_sum += loss
            flag += 1
            if flag >= BATCH_SIZE:
                # バッチサイズで損失を平均して、重みを更新します
                loss = loss_batch_sum / BATCH_SIZE
                loss.backward()
                optimizer.step()
                loss_batch_sum = 0
                flag = 0
            loss_sum += loss
            accuracy_sum += int(category_vector_list[i] == torch.argmax(pred_category_vector[-1][-1]))
        # 正答率と損失の平均を求めます
        accuracy = accuracy_sum / len(words_vector_list)
        print('accuracy(train):', accuracy)
        loss_average = loss_sum / len(words_vector_list)
        print('loss(train):', loss_average)

    # 評価データを読み込みます
    words_vector_list, category_vector_list = chapter9_82.make_dataset('../100knock_chapter6/test.feature.txt', word_id_dict, category_table)

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
    # accuracy(train): 0.47
    # loss(train): tensor(1.1607, grad_fn=<DivBackward0>)
    # accuracy(train): 0.752
    # loss(train): tensor(0.7453, grad_fn=<DivBackward0>)
    # accuracy(train): 0.953
    # loss(train): tensor(0.2088, grad_fn=<DivBackward0>)

    # mini-test.feature.txtでのテスト結果（500文）
    # accuracy(test): 0.524
    # loss(test): tensor(1.3581, grad_fn=<DivBackward0>)

if __name__ == '__main__':
    main()