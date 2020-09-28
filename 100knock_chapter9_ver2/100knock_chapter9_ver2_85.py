# -*- coding: utf-8 -*-

import sys
import os
import torch
from torch import nn, optim
import utils
import models

"""
TensorDataset, DataLoader を使わない場合
（pack_padded_sequence により PackedSequence を使います）
"""

def main():
    # コマンドライン引数から必要な情報を得ます
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    if len(sys.argv) < 5:
        print('python 100knock_chapter9_ver2_80 [dict_filepath] [train_filepath] [batch_size] [num_layers]')
        sys.exit()
    filepath = sys.argv[1]
    train_filepath = sys.argv[2]
    batch_size = sys.argv[3]
    num_layers = sys.argv[4]
    while (1):
        if os.path.exists(filepath) and os.path.exists(train_filepath) and batch_size.isdigit() and num_layers.isdigit():
            word_id_dict = utils.make_word_id_dict(filepath)
            train_category, train_title = utils.split_category_title(train_filepath, category_table, word_id_dict)
            break
        if not os.path.exists(filepath):
            print('対象の辞書となるファイルが存在しません。')
            print('もう一度、入力して下さい。')
            filepath = input()
        if not os.path.exists(train_filepath):
            print('訓練データの対象ファイルが存在しません。')
            print('もう一度、入力して下さい。')
            train_filepath = input()
        if not batch_size.isdigit():
            print('バッチサイズが正しくありません。')
            print('学習時のバッチサイズを入力して下さい。（整数のみ）')
            batch_size = input()
        if not num_layers.isdigit():
            print('レイヤー数が正しくありません。')
            print('モデルのレイヤー数を入力して下さい。（整数のみ）')
            num_layers = input()
    batch_size = int(batch_size)
    num_layers = int(num_layers)

    # MultiRNNモデルを作ります
    input_size = max(word_id_dict.values()) + 1
    embed_size = 300
    hidden_size = 50
    output_size = 4
    model = models.MultiRNN_simple_minibatch(num_layers, input_size, embed_size, hidden_size, output_size)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 訓練を行います
    epoch_num = 10

    # バッチサイズごとに訓練データを区切ります（余りは ramain として最後のバッチとして加えます）
    batch_num = int(len(train_title) / batch_size)
    remain_title = train_title[batch_num * batch_size - 1:]
    remain_len_title = torch.tensor([len(title) for title in remain_title])
    remain_title = sorted(remain_title, key=lambda x: x.shape[0], reverse=True)
    remain_title = nn.utils.rnn.pad_sequence(remain_title, batch_first=True)
    remain_len_title , sort_idx = torch.sort(remain_len_title, descending=True)
    remain_category = train_category[batch_num * batch_size - 1:]
    remain_category = torch.gather(remain_category, -1, sort_idx)

    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_accuracy = 0
        for i in range(batch_num):
            optimizer.zero_grad()
            model.reset(batch_size=batch_size)

            # 訓練データからバッチサイズごとに取り出します
            batch_title = train_title[i * batch_size : (i+1) * batch_size]
            len_title = torch.tensor([len(title) for title in batch_title])
            batch_title = sorted(batch_title, key=lambda x: x.shape[0], reverse=True)
            batch_title = nn.utils.rnn.pad_sequence(batch_title, batch_first=True)
            len_title, sort_idx = torch.sort(len_title, descending=True)
            true_category = train_category[i * batch_size : (i+1) * batch_size]
            true_category = torch.gather(true_category, -1, sort_idx)

            # 予測・損失の計算・勾配の更新を行います
            pred_train_category = model.forward(batch_title, len_title)
            train_loss = loss(pred_train_category, true_category)
            train_loss.backward()
            optimizer.step()
            epoch_loss.append(train_loss.item())
            epoch_accuracy += (true_category == pred_train_category.argmax(1)).sum().item()

        # remain での訓練を行います
        optimizer.zero_grad()
        model.reset(batch_size=len(remain_title))

        pred_train_category = model.forward(remain_title, remain_len_title)
        train_loss = loss(pred_train_category, remain_category)
        train_loss.backward()
        optimizer.step()
        epoch_loss.append(train_loss.item())
        epoch_accuracy += (remain_category == pred_train_category.argmax(1)).sum().item()

        average_loss = sum(epoch_loss) / len(epoch_loss)
        accuracy_rate = epoch_accuracy / len(train_category)
        print('損失：', average_loss, '\t正解率：', accuracy_rate)
if __name__ == '__main__':
    main()

"""
<実行結果>

損失： 9.98419209341438         正解率： 0.4546050168476226
損失： 9.972492895221123        正解率： 0.4816548109322351
損失： 9.963029909083312        正解率： 0.48362036690378135
損失： 9.918666790588702        正解率： 0.4860539123923624
損失： 9.975430078648984        正解率： 0.4855859228753276
損失： 9.96883039312312         正解率： 0.48025084238113064
損失： 9.845703159430885        正解率： 0.4901722201422688
損失： 9.949607559781514        正解率： 0.484556345937851
損失： 9.790772286388941        正解率： 0.4920441782104081
損失： 9.708938243451916        正解率： 0.49522650692624487
"""