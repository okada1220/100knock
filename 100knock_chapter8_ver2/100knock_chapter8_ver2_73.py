import os
import torch
from torch import nn, optim
import models

def main():
    # ファイル名を入力
    print('対象ファイルへのパスを入力して下さい。')
    filepath = input()

    if os.path.exists(filepath):  # 例外処理
        # 特徴行列Ｘと正解ベクトルＹを得ます
        x_matrix, y_vector = torch.load(filepath).values()

        # 単層ニューラルネットワークモデルを作ります
        INPUT_SIZE = 300
        OUTPUT_SIZE = 4
        singlenet_model = models.SingleNet_simple(INPUT_SIZE, OUTPUT_SIZE)
        loss = nn.CrossEntropyLoss()  # 損失関数を設定します
        optimizer = optim.SGD(singlenet_model.parameters(), lr=0.1)  # オプティマイザを設定します

        # 学習の前後で損失が減少しているか確認します（前）
        pred_y_vector = singlenet_model.forward(x_matrix[0])
        l_loss = loss(pred_y_vector.unsqueeze(0), y_vector[0].unsqueeze(0))
        print(l_loss)

        # 学習を行います
        EPOCH_NUM = 100
        for epoch in range(EPOCH_NUM):
            for i, x_vector in enumerate(x_matrix):
                optimizer.zero_grad()
                pred_y_vector = singlenet_model.forward(x_vector)
                l_loss = loss(pred_y_vector.unsqueeze(0), y_vector[i].unsqueeze(0))
                l_loss.backward()
                optimizer.step()

        # 学習の前後で損失が減少しているか確認します（後）
        pred_y_vector = singlenet_model.forward(x_matrix[0])
        l_loss = loss(pred_y_vector.unsqueeze(0), y_vector[0].unsqueeze(0))
        print(l_loss)

        # 学習後のパラメータを表示、保存します
        parameters = singlenet_model.parameters()
        for parameter in parameters:
            print(parameter)
            w_matrix = parameter.detach()
            torch.save({'w_matrix':w_matrix},
                       'w_matrix.pth')
    else:
        print('対象ファイルが存在しません。\n')

if __name__ == '__main__':
    main()

"""
<実行結果>

対象ファイルへのパスを入力して下さい。
train_vectorizer.pth
tensor(1.4577, grad_fn=<NllLossBackward>)
tensor(0.0079, grad_fn=<NllLossBackward>)
Parameter containing:
tensor([[-2.1218,  0.5580,  1.7134,  ...,  0.0501,  1.7259, -1.5810],
        [ 0.5895,  0.6832,  1.8614,  ...,  0.1481, -3.6026, -1.7433],
        [-0.0786,  1.4271, -5.4061,  ..., -0.5885,  1.7748,  0.8150],
        [ 1.5831, -2.5905,  1.7651,  ...,  0.3774,  0.2317,  2.5443]],
       requires_grad=True)

Process finished with exit code 0
"""