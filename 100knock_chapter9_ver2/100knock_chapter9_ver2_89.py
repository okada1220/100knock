# -*- coding: utf-8 -*-

from transformers import *
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import argparse
import tqdm
import models

# 前処理を行うクラスです
class MyPreproccesing(object):
    def __init__(self, tokenizer, category_table):
        self.tokenizer = tokenizer
        self.category_table = category_table

    # ファイルからタイトルとカテゴリをそれぞれＩＤ化します
    def trans_file(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            categories = []
            titles = []
            for line in file:
                category, title = line.split('\t')
                categories.append(self.category_table[category])
                title = self.tokenizer.encode(title)
                title_info = (title, len(title))
                titles.append(title_info)
        return categories, titles

# データセットを作るクラスです
class MyDataset(Dataset):
    def __init__(self, categories, titles):
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
    max_len = max([title_len for title, title_len in title_info])
    titles = []
    attention_masks = []
    for title, title_len in title_info:
        pad_title = torch.cat([torch.tensor(title), torch.zeros(max_len - title_len, dtype=torch.long)], dim=-1)
        titles.append(pad_title)
        attention_mask = torch.tensor([1] * title_len + [0] * (max_len - title_len))
        attention_masks.append(attention_mask)
    category = torch.tensor(category)
    titles = torch.stack(titles)
    attention_masks = torch.stack(attention_masks)
    return titles, attention_masks, category

def main():
    # コマンドライン引数から必要な情報を得ます
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='python 100knock_chapter9_ver2_89 [train_filepath]')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    # 前処理に必要なものを用意します
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    # 前処理、データセット、データローダを作ります
    mypreprocessing = MyPreproccesing(tokenizer, category_table)
    categories, titles = mypreprocessing.trans_file(args.filepath)
    train_dataset = MyDataset(categories, titles)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=pad_collate)
    model = models.BertClassifier()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 訓練を行います
    epoch_num = 10
    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_accuracy = 0
        model.train()
        print('Epoch:', epoch+1)
        for title, title_len, category in tqdm.tqdm(train_dataloader):
            optimizer.zero_grad()
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

(訓練データにmini-train.txt(1000事例)を利用(batch=16, lr=0.01)
Epoch: 1
100%|█████████████████████████████████████████████████████| 63/63 [10:26<00:00,  9.94s/it]
損失： 2.5799834179499794       正解率： 0.38
Epoch: 2
100%|█████████████████████████████████████████████████████| 63/63 [12:58<00:00, 12.36s/it]
損失： 1.5509338833036876       正解率： 0.393
Epoch: 3
100%|█████████████████████████████████████████████████████| 63/63 [13:09<00:00, 12.54s/it]
損失： 1.666892075349414        正解率： 0.362
Epoch: 4
100%|█████████████████████████████████████████████████████| 63/63 [13:34<00:00, 12.93s/it]
損失： 1.6362449347026764       正解率： 0.374
Epoch: 5
100%|█████████████████████████████████████████████████████| 63/63 [12:59<00:00, 12.37s/it]
損失： 1.6335519769835094       正解率： 0.387
Epoch: 6
100%|█████████████████████████████████████████████████████| 63/63 [12:57<00:00, 12.34s/it]
損失： 1.5906686858525352       正解率： 0.373
Epoch: 7
100%|█████████████████████████████████████████████████████| 63/63 [13:06<00:00, 12.48s/it]
損失： 1.649232602308667        正解率： 0.356
Epoch: 8
100%|█████████████████████████████████████████████████████| 63/63 [13:15<00:00, 12.63s/it]
損失： 1.9644524133394634       正解率： 0.36
Epoch: 9
100%|█████████████████████████████████████████████████████| 63/63 [13:16<00:00, 12.65s/it]
損失： 1.840375544532897        正解率： 0.392
Epoch: 10
100%|█████████████████████████████████████████████████████| 63/63 [13:07<00:00, 12.50s/it]
損失： 1.802176937224373        正解率： 0.38

(訓練データにtrain.txt(約10000事例)を利用(batch=32, lr=0.01))
Epoch: 1
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [1:35:28<00:00, 17.15s/it]
損失： 1.563155860244157        正解率： 0.3948895544739798
Epoch: 2
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [1:46:47<00:00, 19.18s/it]
損失： 1.6133469762559423       正解率： 0.3852489704230625
Epoch: 3
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [2:12:25<00:00, 23.79s/it]
損失： 1.4641170046643583       正解率： 0.38393859977536504
Epoch: 4
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [1:46:26<00:00, 19.12s/it]
損失： 1.4876200965421642       正解率： 0.39086484462748033
Epoch: 5
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [1:41:11<00:00, 18.18s/it]
損失： 1.4842760905534207       正解率： 0.38880569075252713
Epoch: 6
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [1:40:56<00:00, 18.13s/it]
損失： 1.447159011385398        正解率： 0.39648071883189817
Epoch: 7
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [1:39:47<00:00, 17.93s/it]
損失： 1.5100286338857547       正解率： 0.3813178584799701
Epoch: 8
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [1:41:45<00:00, 18.28s/it]
損失： 1.510240693292218        正解率： 0.38871209284912017
Epoch: 9
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [1:43:23<00:00, 18.57s/it]
損失： 1.5646467546146072       正解率： 0.3941407712467241
Epoch: 10
100%|██████████████████████████████████████████████████████████████████████████████| 334/334 [1:46:00<00:00, 19.04s/it]
損失： 1.5034492652930185       正解率： 0.38665293897416697
"""