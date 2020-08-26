import os
import torch
from torch import nn, optim
import models
from matplotlib import pyplot as plt
import japanize_matplotlib

def main():
    # ファイル名を入力
    print('訓練データファイルへのパスを入力して下さい。')
    train_filepath = input()
    print('検証データファイルへのパスを入力して下さい。')
    valid_filepath = input()
    print('ニューラルモデルの層数を入力して下さい。（１以上の整数）')
    layer_num = int(input())

    if os.path.exists(train_filepath) and os.path.exists(valid_filepath):  # 例外処理
        # 特徴行列Ｘと正解ベクトルＹを得ます
        train_x_matrix, train_y_vector = torch.load(train_filepath).values()
        valid_x_matrix, valid_y_vector = torch.load(valid_filepath).values()

        # 単層ニューラルネットワークモデルを作ります
        INPUT_SIZE = 300
        OUTPUT_SIZE = 4
        singlenet_model = models.Net_simple(INPUT_SIZE, OUTPUT_SIZE, layer_num - 1)
        loss = nn.CrossEntropyLoss()  # 損失関数を設定します
        optimizer = optim.SGD(singlenet_model.parameters(), lr=0.1)  # オプティマイザを設定します

        # 学習を行います
        EPOCH_NUM = 10
        train_losses = []  # 訓練データの損失
        train_accuracies = []  # 訓練データの正解率
        valid_losses = []  # 検証データの損失
        valid_accuracies = []  # 検証データの正解率
        for epoch in range(EPOCH_NUM):
            epoch_losses = []
            epoch_accuracy = 0
            for i, x_vector in enumerate(train_x_matrix):
                optimizer.zero_grad()
                pred_y_vector = singlenet_model.forward(x_vector)
                l_loss = loss(pred_y_vector.unsqueeze(0), train_y_vector[i].unsqueeze(0))
                l_loss.backward()
                optimizer.step()
                epoch_losses.append(l_loss.item())
                epoch_accuracy += (train_y_vector[i].item() == pred_y_vector.argmax().item())
            # 訓練データでの損失・正解率を計算します
            train_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(train_loss)
            train_accuracy = epoch_accuracy / len(train_x_matrix)
            train_accuracies.append(train_accuracy)

            # 検証データでの損失・正解率を計算します
            pred_y_vector = singlenet_model.forward(valid_x_matrix)
            valid_loss = loss(pred_y_vector, valid_y_vector)
            valid_losses.append(valid_loss)
            valid_accuracy = (valid_y_vector == pred_y_vector.argmax(1)).sum().item() / len(valid_x_matrix)
            valid_accuracies.append(valid_accuracy)

        print('損失グラフを表示します。\n')
        print(train_losses[-1])
        print(valid_losses[-1].item())
        plt.plot(train_losses, label='train_loss', color='k')
        plt.plot(valid_losses, label='valid_loss', color='r')
        plt.show()


        print('正解率グラフを表示します。\n')
        print(train_accuracies[-1])
        print(valid_accuracies[-1])
        plt.plot(train_accuracies, label='train_accuracy', color='k')
        plt.plot(valid_accuracies, label='valid_accuracy', color='r')
        plt.show()
    else:
        print('対象ファイルが存在しません。\n')

if __name__ == '__main__':
    main()

"""
<実行結果>

訓練データファイルへのパスを入力して下さい。
train_vectorizer.pth
検証データファイルへのパスを入力して下さい。
valid_vectorizer.pth
ニューラルモデルの層数を入力して下さい。（１以上の整数）
1
損失グラフを表示します。

0.24678942248165944
0.35046401619911194
正解率グラフを表示します。

0.9145451141894422
0.875

Process finished with exit code 0

訓練データファイルへのパスを入力して下さい。
train_vectorizer.pth
検証データファイルへのパスを入力して下さい。
valid_vectorizer.pth
ニューラルモデルの層数を入力して下さい。（１以上の整数）
2
損失グラフを表示します。

0.030504226114169503
0.5365469455718994
正解率グラフを表示します。

0.9931673530512917
0.8884730538922155

Process finished with exit code 0

訓練データファイルへのパスを入力して下さい。
train_vectorizer.pth
検証データファイルへのパスを入力して下さい。
valid_vectorizer.pth
ニューラルモデルの層数を入力して下さい。（１以上の整数）
3
損失グラフを表示します。

0.09365904485169624
0.7488106489181519
正解率グラフを表示します。

0.9678023212280045
0.8847305389221557

Process finished with exit code 0

訓練データファイルへのパスを入力して下さい。
train_vectorizer.pth
検証データファイルへのパスを入力して下さい。
valid_vectorizer.pth
ニューラルモデルの層数を入力して下さい。（１以上の整数）
4
損失グラフを表示します。

0.12296752724816518
0.656745970249176
正解率グラフを表示します。

0.9573193560464246
0.8832335329341318

Process finished with exit code 0

訓練データファイルへのパスを入力して下さい。
train_vectorizer.pth
検証データファイルへのパスを入力して下さい。
valid_vectorizer.pth
ニューラルモデルの層数を入力して下さい。（１以上の整数）
5
損失グラフを表示します。

0.19389479326027045
0.8506854176521301
正解率グラフを表示します。

0.9362598277798577
0.8712574850299402

Process finished with exit code 0
"""