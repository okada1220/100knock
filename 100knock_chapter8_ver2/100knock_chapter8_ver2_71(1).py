import os
import torch
from torch import nn

"""
ニューラルモデルを使わなかった場合
"""

def main():
    # ファイル名を入力
    print('対象ファイルへのパスを入力して下さい。')
    filepath = input()

    if os.path.exists(filepath):  # 例外処理
        # 特徴行列Ｘと正解ベクトルＹを得ます
        x_matrix, y_vector = torch.load(filepath).values()

        # ランダムを固定した状態で重み行列Ｗを作ります
        torch.manual_seed(0)
        w_matrix = torch.randn(300, 4)

        # 事例 x_1 の予測をします
        pred_y1_vector = torch.matmul(x_matrix[0], w_matrix)  # mv だと上手くいかないです
        # pred_y1_vector = torch.mm(x_matrix[0].unsqueeze(0), w_matrix)  行列に変換して計算した場合（ただし、softmax は dim=1）
        pred_y1_vector = nn.functional.softmax(pred_y1_vector, dim=0)
        print('y1:')
        print(pred_y1_vector)

        # 事例 x_1 ~ x_4 の集合の予測をします
        pred_y_vector = torch.mm(x_matrix[:4], w_matrix)
        pred_y_vector = nn.functional.softmax(pred_y_vector, dim=1)
        print('Y:')
        print(pred_y_vector)
    else:
        print('対象ファイルが存在しません。\n')

if __name__ == '__main__':
    main()

"""
<実行結果>

対象ファイルへのパスを入力して下さい。
train_vectorizer.pth
y1:
tensor([0.1757, 0.4239, 0.2065, 0.1940])
Y:
tensor([[0.1757, 0.4239, 0.2065, 0.1940],
        [0.4047, 0.5529, 0.0213, 0.0211],
        [0.0431, 0.0149, 0.0519, 0.8900],
        [0.0692, 0.8550, 0.0735, 0.0024]])

Process finished with exit code 0
"""