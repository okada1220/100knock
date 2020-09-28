# -*- coding: utf-8 -*-

import sys
import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import utils
import models

"""
TensorDataset, DataLoader を使う場合
"""

def main():
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    if len(sys.argv) < 4:
        print('python 100knock_chapter9_ver2_80 [dict_filepath] [train_filepath] [batch_size]')
        sys.exit()
    filepath = sys.argv[1]
    train_filepath = sys.argv[2]
    batch_size = sys.argv[3]
    while (1):
        if os.path.exists(filepath) and os.path.exists(train_filepath) and batch_size.isdigit():
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
    batch_size = int(batch_size)

    max_length = 0
    for title in train_title:
        if max_length < len(title):
            max_length = len(title)
    padding_id = max(word_id_dict.values()) + 1
    padding_train_title = torch.tensor([[len(title)]
                                        + title.tolist()
                                        + [padding_id] * (max_length - len(title))
                                        for title in train_title])
    train_dataset = TensorDataset(padding_train_title, train_category)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_size = padding_id + 1
    embed_size = 300
    hidden_size = 50
    output_size = 4
    model = models.RNN_simple_minibatch_gpu(input_size, embed_size, hidden_size, output_size, padding_id).cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    epoch_num = 10
    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_accuracy = 0
        for title, category in train_loader:
            optimizer.zero_grad()
            model.reset(batch_size=len(title))
            title = title.cuda()
            category = category.cuda()
            title_len = title[:, 0]
            title_con = title[:, 1:]
            pred_train_category = model.forward(title_con, title_len)
            train_loss = loss(pred_train_category, category)
            train_loss.backward()
            optimizer.step()
            epoch_loss.append(train_loss.item())
            epoch_accuracy += (category == pred_train_category.argmax(1)).sum().item()
        average_loss = sum(epoch_loss) / len(epoch_loss)
        accuracy_rate = epoch_accuracy / len(train_category)
        print('損失：', average_loss, '\t正解率：', accuracy_rate)
if __name__ == '__main__':
    main()
