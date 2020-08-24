import os
import torch
from torch import nn
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
        parameters = singlenet_model.parameters()  # オプティマイザを設定します
        for parameter in parameters:  # 初期パラメータを確認します
            print(parameter)
            print(parameter.grad)

        # 事例 x_1 の予測・損失・勾配を計算します
        pred_y1_vector = singlenet_model.forward(x_matrix[:1])
        print('y1:')
        print(pred_y1_vector)
        l1_loss = loss(pred_y1_vector, y_vector[:1])
        print('l1_loss:')
        print(l1_loss)
        l1_loss.backward()
        print('l1_grad:')
        parameters = singlenet_model.parameters()
        for parameter in parameters:
            print(parameter.grad)

        # 事例 x_1 ~ x_4 の集合の予測・損失・勾配を計算します
        pred_y_matrix = singlenet_model.forward(x_matrix[:4])
        print('Y:')
        print(pred_y_matrix)
        l_loss = loss(pred_y_matrix, y_vector[:4])
        print('L_loss:')
        print(l_loss)
        l_loss.backward()
        print('L_grad:')
        parameters = singlenet_model.parameters()
        for parameter in parameters:
            print(parameter.grad)
    else:
        print('対象ファイルが存在しません。\n')

if __name__ == '__main__':
    main()

"""
<実行結果>

対象ファイルへのパスを入力して下さい。
train_vectorizer.pth
Parameter containing:
tensor([[-0.0004,  0.0310, -0.0475,  ..., -0.0430,  0.0237, -0.0194],
        [-0.0279,  0.0104, -0.0300,  ...,  0.0535,  0.0547,  0.0521],
        [-0.0487, -0.0218, -0.0397,  ..., -0.0037,  0.0193,  0.0173],
        [ 0.0483,  0.0567,  0.0525,  ..., -0.0198,  0.0299, -0.0142]],
       requires_grad=True)
None
y1:
tensor([[ 0.0051, -0.0208, -0.1061, -0.0207]], grad_fn=<MmBackward>)
l1_loss:
tensor(1.4577, grad_fn=<NllLossBackward>)
l1_grad:
tensor([[ 0.0171,  0.0102, -0.0105,  ..., -0.0199, -0.0020,  0.0057],
        [ 0.0167,  0.0100, -0.0103,  ..., -0.0194, -0.0020,  0.0056],
        [-0.0505, -0.0302,  0.0311,  ...,  0.0587,  0.0060, -0.0168],
        [ 0.0167,  0.0100, -0.0103,  ..., -0.0194, -0.0020,  0.0056]])
Y:
tensor([[ 0.0051, -0.0208, -0.1061, -0.0207],
        [-0.0041,  0.0298,  0.0108,  0.0664],
        [-0.0021, -0.0169, -0.0339,  0.0339],
        [-0.0016,  0.0579, -0.0466,  0.1007]], grad_fn=<MmBackward>)
L_loss:
tensor(1.4268, grad_fn=<NllLossBackward>)
L_grad:
tensor([[ 0.0393,  0.0164,  0.0198,  ..., -0.0676, -0.0324,  0.0209],
        [ 0.0174,  0.0089, -0.0242,  ..., -0.0140,  0.0129,  0.0046],
        [-0.0739, -0.0340,  0.0290,  ...,  0.0951,  0.0060, -0.0301],
        [ 0.0172,  0.0087, -0.0246,  ..., -0.0136,  0.0136,  0.0046]])

Process finished with exit code 0
"""