import os
import torch
from torch import nn, optim
import models
from matplotlib import pyplot as plt

def main():
    # ファイル名を入力
    print('訓練データファイルへのパスを入力して下さい。')
    train_filepath = input()
    print('検証データファイルへのパスを入力して下さい。')
    valid_filepath = input()

    if os.path.exists(train_filepath) and os.path.exists(valid_filepath):  # 例外処理
        # 特徴行列Ｘと正解ベクトルＹを得ます
        train_x_matrix, train_y_vector = torch.load(train_filepath).values()
        valid_x_matrix, valid_y_vector = torch.load(valid_filepath).values()

        # 単層ニューラルネットワークモデルを作ります
        INPUT_SIZE = 300
        OUTPUT_SIZE = 4
        singlenet_model = models.SingleNet_simple(INPUT_SIZE, OUTPUT_SIZE)
        loss = nn.CrossEntropyLoss()  # 損失関数を設定します
        optimizer = optim.SGD(singlenet_model.parameters(), lr=0.1)  # オプティマイザを設定します

        # 学習を行います
        EPOCH_NUM = 100
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
        plt.plot(train_losses, label='train_loss')
        plt.plot(valid_losses, label='valid_loss')
        plt.show()

        print('正解率グラフを表示します。\n')
        plt.plot(train_accuracies, label='train_accuracy')
        plt.plot(valid_accuracies, label='valid_accuracy')
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
損失グラフを表示します。

findfont: Font family ['IPAexGothic'] not found. Falling back to DejaVu Sans.
正解率グラフを表示します。


Process finished with exit code 0
"""