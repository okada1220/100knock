import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import time
import models

def main():
    # ファイル名を入力
    print('対象ファイルへのパスを入力して下さい。')
    train_filepath = input()
    print('学習時のバッチサイズを入力して下さい。(整数のみ）')
    batchsize = int(input())

    if os.path.exists(train_filepath):  # 例外処理
        # 特徴行列Ｘと正解ベクトルＹを得ます
        train_x_matrix, train_y_vector = torch.load(train_filepath).values()

        # 訓練データをミニバッチ化します
        train_dataset = TensorDataset(train_x_matrix, train_y_vector)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

        # 単層ニューラルネットワークモデルを作ります
        INPUT_SIZE = 300
        OUTPUT_SIZE = 4
        singlenet_model = models.SingleNet_simple(INPUT_SIZE, OUTPUT_SIZE).cuda()  # GPU上で学習します
        loss = nn.CrossEntropyLoss()  # 損失関数を設定します
        optimizer = optim.SGD(singlenet_model.parameters(), lr=0.1)  # オプティマイザを設定します

        # 学習を行います
        EPOCH_NUM = 10
        for epoch in range(EPOCH_NUM):
            start_time = time.time()
            epoch_accuracy = 0
            for x_vector, y_vector in train_loader:
                optimizer.zero_grad()
                x_vector = x_vector.cuda()  # GPU上で学習します
                y_vector = y_vector.cuda()  # GPU上で学習します
                pred_y_vector = singlenet_model.forward(x_vector)
                l_loss = loss(pred_y_vector, y_vector)
                l_loss.backward()
                optimizer.step()
                epoch_accuracy += (y_vector == pred_y_vector.argmax(1)).sum().item()
            # 正解率・実行時間を計算します
            train_accuracy = epoch_accuracy / len(train_x_matrix)
            end_time = time.time()
            exe_time = end_time - start_time
            print('正解率：', train_accuracy, '\t実行時間：', exe_time)
    else:
        print('対象ファイルが存在しません。\n')
if __name__ == '__main__':
    main()