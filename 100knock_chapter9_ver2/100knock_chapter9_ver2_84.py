# -*- coding: utf-8 -*-

import sys
import os
import gensim
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
        print('python 100knock_chapter9_ver2_80 [dict_filepath] [train_filepath] [weight_filepath] [batch_size]')
        sys.exit()
    dict_filepath = sys.argv[1]
    train_filepath = sys.argv[2]
    weight_filepath = sys.argv[3]
    batch_size = sys.argv[4]
    while (1):
        if os.path.exists(dict_filepath) and os.path.exists(train_filepath) and os.path.exists(weight_filepath) and batch_size.isdigit():
            word_id_dict = utils.make_word_id_dict(dict_filepath)
            train_category, train_title = utils.split_category_title(train_filepath, category_table, word_id_dict)
            break
        if not os.path.exists(dict_filepath):
            print('対象の辞書となるファイルが存在しません。')
            print('もう一度、入力して下さい。')
            dict_filepath = input()
        if not os.path.exists(train_filepath):
            print('訓練データの対象ファイルが存在しません。')
            print('もう一度、入力して下さい。')
            train_filepath = input()
        if not os.path.exists(weight_filepath):
            print('重みを初期化するためのファイルが存在しません。')
            print('もう一度、入力して下さい。')
            weight_filepath = input()
        if not batch_size.isdigit():
            print('バッチサイズが正しくありません。')
            print('学習時のバッチサイズを入力して下さい。（整数のみ）')
            batch_size = input()
    batch_size = int(batch_size)
    weight_file = gensim.models.KeyedVectors.load_word2vec_format(weight_filepath, binary=True)
    weight = weight_file.wv.syn0

    # 重み行列の初期化
    weight_matrix = torch.zeros([max(word_id_dict.values())+1, weight.shape[1]])
    # id = 0 -> 0 ベクトルの重み、id > 0 -> weight の重み
    for word, id in word_id_dict.items():
        if id == 0:
            continue
        if word in weight_file:
            weight_matrix[id] = torch.Tensor(weight_file[word])

    # RNNモデルを作ります
    input_size = max(word_id_dict.values()) + 1
    embed_size = weight.shape[1]
    hidden_size = 50
    output_size = 4
    model = models.RNN_simple_minibatch_2(input_size, embed_size, hidden_size, output_size)
    model.embed.weight = nn.Parameter(weight_matrix)
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
    remain_len_title, sort_idx = torch.sort(remain_len_title, descending=True)
    remain_category = train_category[batch_num * batch_size - 1:]
    remain_category = torch.gather(remain_category, -1, sort_idx)

    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_accuracy = 0
        for i in range(batch_num):
            optimizer.zero_grad()
            model.reset(batch_size=batch_size)

            # 訓練データからバッチサイズごとに取り出します
            batch_title = train_title[i * batch_size: (i + 1) * batch_size]
            len_title = torch.tensor([len(title) for title in batch_title])
            batch_title = sorted(batch_title, key=lambda x: x.shape[0], reverse=True)
            batch_title = nn.utils.rnn.pad_sequence(batch_title, batch_first=True)
            len_title, sort_idx = torch.sort(len_title, descending=True)
            true_category = train_category[i * batch_size: (i + 1) * batch_size]
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

100knock_chapter9_ver2_84.py:48: DeprecationWarning: Call to deprecated `wv` (Attribute wil
l be removed in 4.0.0, use self instead).
  weight = weight_file.wv.syn0
100knock_chapter9_ver2_84.py:48: DeprecationWarning: Call to deprecated `syn0` (Attribute w
ill be removed in 4.0.0, use self.vectors instead).
  weight = weight_file.wv.syn0
..\torch\csrc\utils\tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeab
le, and PyTorch does not support non-writeable tensors. This means you can write to the und
erlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the a
rray to protect its data or make it writeable before converting it to a tensor. This type o
f warning will be suppressed for the rest of this program.
損失： 2.902438108283086        正解率： 0.40612130288281545
損失： 2.8289654159171644       正解率： 0.4378509921377761
損失： 2.631089495624175        正解率： 0.46845750655185325
損失： 2.3702439151667805       正解率： 0.5248970423062523
損失： 2.2351310133625115       正解率： 0.5497004867090978
損失： 2.302043345892171        正解率： 0.5378135529764133
損失： 2.3786414648842076       正解率： 0.5199363534256832
損失： 2.4012581266808284       正解率： 0.5124485211531262
損失： 2.397447719930461        正解率： 0.5171284163234744
損失： 2.403401160743212        正解率： 0.5147884687383003
"""