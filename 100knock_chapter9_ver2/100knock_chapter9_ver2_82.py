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
        print('python 100knock_chapter9_ver2_80 [dict_filepath] [train_filepath] [test_filepath]')
        sys.exit()
    filepath = sys.argv[1]
    train_filepath = sys.argv[2]
    test_filepath = sys.argv[3]
    while(1):
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

    # RNNモデルを作ります
    input_size = max(word_id_dict.values()) + 1
    embed_size = 300
    hidden_size = 50
    output_size = 4
    model = models.RNN_simple(input_size, embed_size, hidden_size, output_size)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 訓練を行います
    epoch_num = 10
    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_accuracy = 0
        for i, title in enumerate(train_title):
            optimizer.zero_grad()
            model.reset()
            pred_train_category = model.forward(title.unsqueeze(0)).squeeze(0)
            train_loss = loss(pred_train_category.unsqueeze(0), train_category[i].unsqueeze(0))
            train_loss.backward()
            optimizer.step()
            epoch_loss.append(train_loss.item())
            epoch_accuracy += (train_category[i].item() == pred_train_category.argmax().item())
        average_loss = sum(epoch_loss) / len(epoch_loss)
        accuracy_rate = epoch_accuracy / len(train_category)
        print('【訓練データ】')
        print('損失：', average_loss, '\t正解率：', accuracy_rate)

        pred_test_category = torch.stack([model.forward(feature.unsqueeze(0)).squeeze(0) for feature in test_title])
        test_loss = loss(pred_test_category, test_category)
        test_accuracy = (test_category == pred_test_category.argmax(1)).sum().item() / len(test_category)
        print('【評価データ】')
        print('損失：', test_loss.item(), '\t正解率：', test_accuracy)
if __name__ == '__main__':
    main()

"""
<実行結果>

【訓練データ】
損失： 2.708293529479509        正解率： 0.43026956196181204
【評価データ】
損失： 2.4700915813446045       正解率： 0.4977544910179641
【訓練データ】
損失： 2.7583949048160274       正解率： 0.43485585922875325
【評価データ】
損失： 3.2070472240448  正解率： 0.36676646706586824
【訓練データ】
損失： 2.7371164810736834       正解率： 0.44374766005241484
【評価データ】
損失： 2.893486261367798        正解率： 0.406437125748503
【訓練データ】
損失： 2.666560555579879        正解率： 0.46050168476226133
【評価データ】
損失： 2.254788398742676        正解率： 0.5097305389221557
【訓練データ】
損失： 2.6074613346382725       正解率： 0.47210782478472485
【評価データ】
損失： 2.742755174636841        正解率： 0.5022455089820359
【訓練データ】
損失： 2.5870205279570815       正解率： 0.4788468738300262
【評価データ】
損失： 3.483412265777588        正解率： 0.4184131736526946
【訓練データ】
損失： 2.549919589207508        正解率： 0.4853051291651067
【評価データ】
損失： 3.180032968521118        正解率： 0.42664670658682635
【訓練データ】
損失： 2.481482737814284        正解率： 0.5001871958068139
【評価データ】
損失： 3.279291868209839        正解率： 0.4161676646706587
【訓練データ】
損失： 2.466070503260794        正解率： 0.5042119056533134
【評価データ】
損失： 2.89460825920105         正解率： 0.39146706586826346
【訓練データ】
損失： 2.413488366119423        正解率： 0.5176900037439162
【評価データ】
損失： 3.356476068496704        正解率： 0.3997005988023952

"""