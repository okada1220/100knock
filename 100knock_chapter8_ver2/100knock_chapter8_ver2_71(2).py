import os
import torch
import models

"""
ニューラルモデルを使った場合
"""

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
        singlenet_model = models.SingleNet(INPUT_SIZE, OUTPUT_SIZE)

        # 事例 x_1 の予測をします
        pred_y1_vector = singlenet_model.forward(x_matrix[:1])
        print('y1:')
        print(pred_y1_vector)

        # 事例 x_1 ~ x_4 の集合の予測をします
        pred_y_matrix = singlenet_model.forward(x_matrix[:4])
        print('Y:')
        print(pred_y_matrix)
    else:
        print('対象ファイルが存在しません。\n')

if __name__ == '__main__':
    main()

"""
<実行結果>

対象ファイルへのパスを入力して下さい。
train_vectorizer.pth
y1:
tensor([[0.2602, 0.2535, 0.2328, 0.2536]], grad_fn=<SoftmaxBackward>)
Y:
tensor([[0.2602, 0.2535, 0.2328, 0.2536],
        [0.2426, 0.2509, 0.2462, 0.2603],
        [0.2506, 0.2469, 0.2427, 0.2598],
        [0.2424, 0.2573, 0.2318, 0.2685]], grad_fn=<SoftmaxBackward>)

Process finished with exit code 0
"""