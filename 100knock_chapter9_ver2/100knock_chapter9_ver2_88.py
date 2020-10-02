# -*- coding: utf-8 -*-

import sys
import os
import torch
from torch import nn, optim
import optuna
import utils
import models

# パラメータ（オプティマイザの種類、学習レート）を受け取り、CNNモデルを作る関数です。モデル、損失関数、オプティマイザを返します。
def set_params(params):
    input_size = max(word_id_dict.values()) + 1
    embed_size = 300
    hidden_size = 50
    output_size = 4
    model = models.MultiNet_simple(input_size, embed_size, hidden_size, output_size)
    loss = nn.CrossEntropyLoss()
    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
    elif params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    return model, loss, optimizer

# 訓練を行う関数です。訓練中の評価データに対する最小の損失の値を返します。
def train_eval_function(model, loss, optimizer, result_print=False):
    epoch_num = 10
    min_test_loss = 9999
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

        pred_test_category = torch.stack([model.forward(feature.unsqueeze(0)) for feature in test_title])
        test_loss = loss(pred_test_category, test_category)
        if min_test_loss > test_loss.item():
            min_test_loss = test_loss.item()

        if result_print == True:
            average_loss = sum(epoch_loss) / len(epoch_loss)
            accuracy_rate = epoch_accuracy / len(train_category)
            print('【訓練データ】')
            print('損失：', average_loss, '\t正解率：', accuracy_rate)
            test_accuracy = (test_category == pred_test_category.argmax(1)).sum().item() / len(test_category)
            print('【評価データ】')
            print('損失：', test_loss.item(), '\t正解率：', test_accuracy)
    return min_test_loss

# optuna 用パラメータ関数です。オプティマイザの種類、学習レートに対して、グリッドサーチを行います。最小化する対象は評価データの最小の損失の値です。
def objective(trial):
    params = {'optimizer': trial.suggest_categorical('optimizer', ['SGD', 'Adam']),
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)}
    model, loss, optimizer = set_params(params)
    min_test_loss = train_eval_function(model, loss, optimizer)
    return min_test_loss

def main():
    # コマンドライン引数から必要な情報を得ます
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    if len(sys.argv) < 4:
        print('python 100knock_chapter9_ver2_88 [dict_filepath] [train_filepath] [test_filepath]')
        sys.exit()
    filepath = sys.argv[1]
    train_filepath = sys.argv[2]
    test_filepath = sys.argv[3]
    while (1):
        if os.path.exists(filepath) and os.path.exists(train_filepath) and os.path.exists(test_filepath):
            global word_id_dict, train_category, train_title, test_category, test_title
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

    # グリッドサーチを行います
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print(best_params)

    # CNNモデルを作ります
    model, loss, optimizer = set_params(best_params)
    train_eval_function(model, loss, optimizer, result_print=True)

if __name__ == '__main__':
    main()

"""
<実行結果>
(minimini-train.txt と minimini-test.txt を利用(共に100事例))

(100knock) C:\Users\naoki\Documents\100本ノック\100knock\100knock_chapter9_ver2>bash test.sh 88
[I 2020-09-28 05:47:22,235] A new study created in memory with name: no-name-e5616bb
4-a559-48a7-b34a-3d74f9ebb690
[I 2020-09-28 05:48:28,665] Trial 0 finished with value: 1.179473876953125 and param
eters: {'optimizer': 'Adam', 'learning_rate': 0.009715456173431038}. Best is trial 0
 with value: 1.179473876953125.
[I 2020-09-28 05:49:57,129] Trial 1 finished with value: 1.088255763053894 and param
eters: {'optimizer': 'Adam', 'learning_rate': 0.002038677190755639}. Best is trial 1
 with value: 1.088255763053894.
[I 2020-09-28 05:51:09,655] Trial 2 finished with value: 1.0701600313186646 and para
meters: {'optimizer': 'Adam', 'learning_rate': 0.0017549097121694161}. Best is trial
 2 with value: 1.0701600313186646.
[I 2020-09-28 05:52:18,710] Trial 3 finished with value: 1.07390296459198 and parame
ters: {'optimizer': 'Adam', 'learning_rate': 0.00019826030684661272}. Best is trial
2 with value: 1.0701600313186646.
[I 2020-09-28 05:53:29,660] Trial 4 finished with value: 1.1599518060684204 and para
meters: {'optimizer': 'Adam', 'learning_rate': 7.287623816985475e-05}. Best is trial
 2 with value: 1.0701600313186646.
[I 2020-09-28 05:54:37,755] Trial 5 finished with value: 1.134211540222168 and param
eters: {'optimizer': 'Adam', 'learning_rate': 0.003427037734859117}. Best is trial 2
 with value: 1.0701600313186646.
[I 2020-09-28 05:55:01,935] Trial 6 finished with value: 1.3850409984588623 and para
meters: {'optimizer': 'SGD', 'learning_rate': 6.580565269359509e-05}. Best is trial
2 with value: 1.0701600313186646.
[I 2020-09-28 05:55:27,512] Trial 7 finished with value: 1.4096931219100952 and para
meters: {'optimizer': 'SGD', 'learning_rate': 2.9847532693272784e-05}. Best is trial
 2 with value: 1.0701600313186646.
[I 2020-09-28 05:56:44,840] Trial 8 finished with value: 1.071580410003662 and param
eters: {'optimizer': 'Adam', 'learning_rate': 0.000168176914675484}. Best is trial 2
 with value: 1.0701600313186646.
[I 2020-09-28 05:57:09,211] Trial 9 finished with value: 1.6714544296264648 and para
meters: {'optimizer': 'SGD', 'learning_rate': 1.3966835482906645e-05}. Best is trial
 2 with value: 1.0701600313186646.
[I 2020-09-28 05:58:20,055] Trial 10 finished with value: 1.1071889400482178 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0011147047191186977}. Best is tria
l 2 with value: 1.0701600313186646.
[I 2020-09-28 05:59:27,786] Trial 11 finished with value: 1.0771479606628418 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.00044326547971008896}. Best is tri
al 2 with value: 1.0701600313186646.
[I 2020-09-28 06:00:35,715] Trial 12 finished with value: 1.0735547542572021 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.00037333044430268737}. Best is tri
al 2 with value: 1.0701600313186646.
[I 2020-09-28 06:01:38,867] Trial 13 finished with value: 1.152944564819336 and para
meters: {'optimizer': 'Adam', 'learning_rate': 0.008151229563322369}. Best is trial
2 with value: 1.0701600313186646.
[I 2020-09-28 06:02:44,784] Trial 14 finished with value: 1.0122438669204712 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0008728571153433217}. Best is tria
l 14 with value: 1.0122438669204712.
[I 2020-09-28 06:04:06,326] Trial 15 finished with value: 1.0421390533447266 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0009098098978960151}. Best is tria
l 14 with value: 1.0122438669204712.
[I 2020-09-28 06:04:32,585] Trial 16 finished with value: 1.1612510681152344 and par
ameters: {'optimizer': 'SGD', 'learning_rate': 0.0008967554527395707}. Best is trial
 14 with value: 1.0122438669204712.
[I 2020-09-28 06:05:40,791] Trial 17 finished with value: 1.1452544927597046 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.004738197606298975}. Best is trial
 14 with value: 1.0122438669204712.
[I 2020-09-28 06:06:44,221] Trial 18 finished with value: 1.0685539245605469 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0006485465777187777}. Best is tria
l 14 with value: 1.0122438669204712.
[I 2020-09-28 06:07:51,356] Trial 19 finished with value: 1.064405083656311 and para
meters: {'optimizer': 'Adam', 'learning_rate': 0.0002368638757180834}. Best is trial
 14 with value: 1.0122438669204712.
[I 2020-09-28 06:08:55,236] Trial 20 finished with value: 1.0789096355438232 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0018101283191131483}. Best is tria
l 14 with value: 1.0122438669204712.
[I 2020-09-28 06:10:03,451] Trial 21 finished with value: 1.0724934339523315 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.00019862647423147617}. Best is tri
al 14 with value: 1.0122438669204712.
[I 2020-09-28 06:11:07,291] Trial 22 finished with value: 1.037819266319275 and para
meters: {'optimizer': 'Adam', 'learning_rate': 0.0005543687790123379}. Best is trial
 14 with value: 1.0122438669204712.
[I 2020-09-28 06:12:11,226] Trial 23 finished with value: 1.0409924983978271 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0006086124498222586}. Best is tria
l 14 with value: 1.0122438669204712.
[I 2020-09-28 06:13:19,581] Trial 24 finished with value: 1.0562183856964111 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0003996190803249511}. Best is tria
l 14 with value: 1.0122438669204712.
[I 2020-09-28 06:14:26,021] Trial 25 finished with value: 1.1508203744888306 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.00010367256041836072}. Best is tri
al 14 with value: 1.0122438669204712.
[I 2020-09-28 06:14:50,479] Trial 26 finished with value: 1.1723052263259888 and par
ameters: {'optimizer': 'SGD', 'learning_rate': 0.0005705651987812016}. Best is trial
 14 with value: 1.0122438669204712.
[I 2020-09-28 06:15:55,596] Trial 27 finished with value: 1.1819210052490234 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0036521390276904003}. Best is tria
l 14 with value: 1.0122438669204712.
[I 2020-09-28 06:17:08,509] Trial 28 finished with value: 1.1236034631729126 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0012684039258353229}. Best is tria
l 14 with value: 1.0122438669204712.
[I 2020-09-28 06:18:15,225] Trial 29 finished with value: 0.9962079524993896 and par
ameters: {'optimizer': 'Adam', 'learning_rate': 0.0006293029908410333}. Best is tria
l 29 with value: 0.9962079524993896.
{'optimizer': 'Adam', 'learning_rate': 0.0006293029908410333}
【訓練データ】
損失： 1.3374072265625  正解率： 0.39
【評価データ】
損失： 1.177138090133667        正解率： 0.53
【訓練データ】
損失： 0.8882586342096329       正解率： 0.81
【評価データ】
損失： 1.1623033285140991       正解率： 0.52
【訓練データ】
損失： 0.6928912550210953       正解率： 0.85
【評価データ】
損失： 1.1389076709747314       正解率： 0.56
【訓練データ】
損失： 0.4941986580193043       正解率： 0.93
【評価データ】
損失： 1.0891776084899902       正解率： 0.57
【訓練データ】
損失： 0.32248084045946596      正解率： 0.97
【評価データ】
損失： 1.0769555568695068       正解率： 0.57
【訓練データ】
損失： 0.18847317066043615      正解率： 1.0
【評価データ】
損失： 1.0621970891952515       正解率： 0.58
【訓練データ】
損失： 0.1088100466504693       正解率： 1.0
【評価データ】
損失： 1.0475611686706543       正解率： 0.59
【訓練データ】
損失： 0.06751649836078286      正解率： 1.0
【評価データ】
損失： 1.0447380542755127       正解率： 0.59
【訓練データ】
損失： 0.04545376423746347      正解率： 1.0
【評価データ】
損失： 1.041375756263733        正解率： 0.59
【訓練データ】
損失： 0.033074457235634326     正解率： 1.0
【評価データ】
損失： 1.039667010307312        正解率： 0.59
"""