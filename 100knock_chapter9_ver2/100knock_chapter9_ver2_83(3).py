# -*- coding: utf-8 -*-

import sys
import os
import nltk
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import utils
import models

"""
Dataset,Dataloader,pad_sequence,pack_padded_sequence等を利用した場合（完成版）
"""

# 前処理を行うクラスです
class MyPreprocessing(object):
    def __init__(self, word_id_dict, category_table):
        self.word_id_dict = word_id_dict
        self.category_table = category_table

    # タイトルのＩＤ系列化
    def trans_title(self, title):
        title = title.strip()
        words = nltk.word_tokenize(title)
        words_id = torch.tensor([self.word_id_dict.get(word, 0) for word in words])
        return words_id

    # カテゴリのＩＤ化
    def trans_category(self, category):
        category_id = torch.tensor([self.category_table[category]])
        return category_id

# データセットを作るクラスです
class MyDataset(Dataset):
    # ファイルからタイトルとカテゴリをそれぞれＩＤ化します
    def __init__(self, filepath, mypreprocessing):
        categories = []
        titles = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                category, title = line.split('\t')
                categories.append(mypreprocessing.trans_category(category))
                title = mypreprocessing.trans_title(title)
                title_info = (title, len(title))
                titles.append(title_info)
        self.data_num = len(titles)
        self.data = titles
        self.label = categories

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        return out_data, out_label

# DataLoader で使うパディング用の関数です
def pad_collate(batch):
    title_info, category = zip(*batch)
    title, title_len = zip(*title_info)
    title = pad_sequence(title, batch_first=True, padding_value=max_id+1)  # パディングします
    title_len = list(title_len)
    category = torch.tensor(category)
    return title, title_len, category

def main():
    # コマンドライン引数から必要な情報を得ます
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    if len(sys.argv) < 4:
        print('python 100knock_chapter9_ver2_83(3) [dict_filepath] [train_filepath] [batch_size]')
        sys.exit()
    filepath = sys.argv[1]
    train_filepath = sys.argv[2]
    batch_size = sys.argv[3]
    while (1):
        if os.path.exists(filepath) and os.path.exists(train_filepath) and batch_size.isdigit():
            word_id_dict = utils.make_word_id_dict(filepath)
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
    global max_id  # 最大ＩＤ
    max_id = max(word_id_dict.values())

    # 前処理、データセット、データローダを作ります
    mypreprocessing = MyPreprocessing(word_id_dict, category_table)
    train_dataset = MyDataset(train_filepath, mypreprocessing)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    # RNNモデルを作ります
    input_size = (max_id+1) + 1
    embed_size = 300
    hidden_size = 50
    output_size = 4
    model = models.RNN_simple_minibatch(input_size, embed_size, hidden_size, output_size, padding_idx=max_id+1)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 訓練を行います
    epoch_num = 10
    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_accuracy = 0
        for title, title_len, category in train_loader:
            optimizer.zero_grad()
            model.reset(batch_size)
            pred_category = model.forward(title, title_len)
            train_loss = loss(pred_category, category)
            train_loss.backward()
            optimizer.step()
            epoch_loss.append(train_loss.item())
            epoch_accuracy += (category == pred_category.argmax(1)).sum().item()
        average_loss = sum(epoch_loss) / len(epoch_loss)
        accuracy_rate = epoch_accuracy / len(train_dataset)
        print('損失：', average_loss, '\t正解率：', accuracy_rate)

if __name__ == '__main__':
    main()

"""
<実行結果>

バッチサイズ：1
損失： 2.75073950391627         正解率： 0.41997379258704604
損失： 2.7584006166177315       正解率： 0.438131785847997
損失： 2.783128717758349        正解率： 0.43335829277424187
損失： 2.74810505539024         正解率： 0.4415949082740547
損失： 2.6802556742855397       正解率： 0.45544739797828526
損失： 2.6501492070283663       正解率： 0.4645263946087608
損失： 2.6422315730604895       正解率： 0.46621115687008613
損失： 2.5827692343178996       正解率： 0.47791089479595655
損失： 2.582783661731196        正解率： 0.47772369898914263
損失： 2.5365016194321726       正解率： 0.4899850243354549
バッチサイズ：2
損失： 1.5108732341204933       正解率： 0.49522650692624487
損失： 1.4287537497008298       正解率： 0.5483901160614002
損失： 1.3682334144230441       正解率： 0.5733807562710596
損失： 1.3389264941717178       正解率： 0.5859228753275927
損失： 1.405558739219938        正解率： 0.5639273680269562
損失： 1.4745720522894024       正解率： 0.5369711718457506
損失： 1.482567177916729        正解率： 0.5325720703856234
損失： 1.45226195950413         正解率： 0.5436166229876451
損失： 1.4499096847723882       正解率： 0.5484837139648072
損失： 1.4565374192116685       正解率： 0.5396855110445526
バッチサイズ：4
損失： 1.1168846239390577       正解率： 0.5894795956570573
損失： 0.9797296225236612       正解率： 0.6607076001497566
損失： 0.9513265233914372       正解率： 0.6670722575814302
損失： 0.8933059773447974       正解率： 0.695806813927368
損失： 0.844131301082036        正解率： 0.7101272931486334
損失： 0.8126662209924995       正解率： 0.7196742792961438
損失： 0.7983223678514875       正解率： 0.7249157618869337
損失： 0.7721849983702412       正解率： 0.7339947585174093
損失： 0.8207216547500135       正解率： 0.7176151254211905
損失： 0.7847053757493877       正解率： 0.7287532759266192
バッチサイズ：8
損失： 1.0058252512906751       正解率： 0.626357169599401
損失： 0.823379023970958        正解率： 0.7115312616997379
損失： 0.7246388916601009       正解率： 0.744852115312617
損失： 0.6664064547164131       正解率： 0.7631973043803819
損失： 0.6197309273442889       正解率： 0.7735866716585549
損失： 0.573057738447863        正解率： 0.7940846125046799
損失： 0.5428404137994091       正解率： 0.8040059902658181
損失： 0.5365167325306796       正解率： 0.8056907525271434
損失： 0.5762508990923445       正解率： 0.7908086858854362
損失： 0.5089731239190906       正解率： 0.816828903032572
"""