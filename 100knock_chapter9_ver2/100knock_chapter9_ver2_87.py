# -*- coding: utf-8 -*-

import sys
import os
import torch
from torch import nn, optim
import utils
import models

def main():
    # コマンドライン引数から必要な情報を得ます
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    if len(sys.argv) < 4:
        print('python 100knock_chapter9_ver2_87 [dict_filepath] [train_filepath] [test_filepath]')
        sys.exit()
    filepath = sys.argv[1]
    train_filepath = sys.argv[2]
    test_filepath = sys.argv[3]
    while (1):
        if os.path.exists(filepath) and os.path.exists(train_filepath) and os.path.exists(test_filepath):
            word_id_dict = utils.make_word_id_dict(filepath)
            train_category, train_title = utils.split_category_title(train_filepath, category_table, word_id_dict)
            test_category, test_title = utils.split_category_title(test_filepath, category_table, word_id_dict)
            break
        if not os.path.exists(filepath):
            print('対象の辞書となるファイルが存在しません。')
            print('もう一度、入力して下さい。')
            filepath = input()
        if not os.path.exists(train_filepath):
            print('訓練データの対象ファイルが存在しません。')
            print('もう一度、入力して下さい。')
            train_filepath = input()
        if not os.path.exists(test_filepath):
            print('評価データの対象ファイルが存在しません。')
            print('もう一度、入力して下さい。')
            test_filepath = input()

    # CNNモデルを作ります
    input_size = max(word_id_dict.values()) + 1
    embed_size = 300
    hidden_size = 50
    output_size = 4
    model = models.CNN_simple(input_size, embed_size, hidden_size, output_size)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 訓練を行います
    epoch_num = 10
    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_accuracy = 0
        for i, title in enumerate(train_title):
            optimizer.zero_grad()
            pred_train_category = model.forward(title.unsqueeze(0))
            train_loss = loss(pred_train_category.unsqueeze(0), train_category[i].unsqueeze(0))
            train_loss.backward()
            optimizer.step()
            epoch_loss.append(train_loss.item())
            epoch_accuracy += (train_category[i].item() == pred_train_category.argmax().item())
        average_loss = sum(epoch_loss) / len(epoch_loss)
        accuracy_rate = epoch_accuracy / len(train_category)
        print('【訓練データ】')
        print('損失：', average_loss, '\t正解率：', accuracy_rate)

        pred_test_category = torch.stack([model.forward(feature.unsqueeze(0)) for feature in test_title])
        test_loss = loss(pred_test_category, test_category)
        test_accuracy = (test_category == pred_test_category.argmax(1)).sum().item() / len(test_category)
        print('【評価データ】')
        print('損失：', test_loss.item(), '\t正解率：', test_accuracy)
if __name__ == '__main__':
    main()

"""
<実行結果>

【訓練データ】
損失： 2.9722532035259412       正解率： 0.3843129913889929
【評価データ】
損失： 2.896604537963867        正解率： 0.5194610778443114
【訓練データ】
損失： 2.678707849817693        正解率： 0.4577873455634594
【評価データ】
損失： 2.1506550312042236       正解率： 0.5688622754491018
【訓練データ】
損失： 2.401152171541447        正解率： 0.5140396855110445
【評価データ】
損失： 2.1303000450134277       正解率： 0.6190119760479041
【訓練データ】
損失： 2.2209864436726994       正解率： 0.5507300636465743
【評価データ】
損失： 1.8765711784362793       正解率： 0.686377245508982
【訓練データ】
損失： 2.1005987552598917       正解率： 0.574691126918757
【評価データ】
損失： 1.8295854330062866       正解率： 0.7020958083832335
【訓練データ】
損失： 1.962843845834062        正解率： 0.6034256832646949
【評価データ】
損失： 1.8398200273513794       正解率： 0.7065868263473054
【訓練データ】
損失： 1.920027711033002        正解率： 0.6111007113440659
【評価データ】
損失： 1.7817037105560303       正解率： 0.688622754491018
【訓練データ】
損失： 1.84763531733994         正解率： 0.6271995507300636
【評価データ】
損失： 1.766423225402832        正解率： 0.7065868263473054
【訓練データ】
損失： 1.8226321612147838       正解率： 0.6298202920254586
【評価データ】
損失： 2.0163049697875977       正解率： 0.6983532934131736
【訓練データ】
損失： 1.8606530049292742       正解率： 0.6249532010482965
【評価データ】
損失： 1.8922873735427856       正解率： 0.7103293413173652
"""