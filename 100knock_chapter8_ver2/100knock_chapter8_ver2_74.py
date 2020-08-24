import os
import torch
from torch import nn
import models

def main():
    # ファイル名を入力
    print('対象ファイルへのパスを入力して下さい。')
    filepath = input()

    while(filepath != ''):  # 連続処理
        if os.path.exists(filepath):  # 例外処理
            # 特徴行列Ｘと正解ベクトルＹ、学習後のパラメータを得ます
            x_matrix, y_vector = torch.load(filepath).values()
            w_matrix = torch.load('w_matrix.pth')['w_matrix']

            # 単層ニューラルネットワークモデルを作ります
            INPUT_SIZE = 300
            OUTPUT_SIZE = 4
            singlenet_model = models.SingleNet(INPUT_SIZE, OUTPUT_SIZE)
            singlenet_model.input.weight = nn.Parameter(w_matrix)  # 初期パラメータを設定します

            # 正解率を求めます
            pred_y_vector = singlenet_model.forward(x_matrix)
            pred_y_vector = pred_y_vector.argmax(1)
            print(pred_y_vector)
            print(y_vector)
            accuracy = (y_vector == pred_y_vector).sum().item() / len(y_vector)
            print('accuracy:', accuracy, '\n')
        else:
            print('対象ファイルが存在しません。\n')
        print('次の対象ファイルへのパスを入力して下さい。（終了：Enter）')  # 連続して処理を行います
        filepath = input()

if __name__ == '__main__':
    main()

"""
<実行結果>

対象ファイルへのパスを入力して下さい。
train_vectorizer.pth
tensor([2, 0, 2,  ..., 2, 0, 2], grad_fn=<NotImplemented>)
tensor([2, 0, 2,  ..., 2, 0, 2])
accuracy: 0.9247472856608012 

次の対象ファイルへのパスを入力して下さい。（終了：Enter）
valid_vectorizer.pth
tensor([0, 2, 0,  ..., 2, 0, 2], grad_fn=<NotImplemented>)
tensor([0, 2, 0,  ..., 2, 0, 2])
accuracy: 0.8705089820359282 

次の対象ファイルへのパスを入力して下さい。（終了：Enter）


Process finished with exit code 0
"""