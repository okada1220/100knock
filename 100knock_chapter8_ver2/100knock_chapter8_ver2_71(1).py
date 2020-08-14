import os
import torch
from torch import nn
import utils

def main():
    print('対象ファイルへのパスを入力して下さい。')
    filepath = input()

    while(filepath != ''):
        if os.path.exists(filepath):
            x_matrix, y_vector = utils.load_vectorizer_file(filepath)
            x_matrix = torch.from_numpy(x_matrix)
            y_vector = torch.from_numpy(y_vector)

            torch.manual_seed(0)
            w_matrix = torch.randn(300, 4)

            pred_y1_vector = torch.matmul(x_matrix[0], w_matrix)  # mv だと上手くいかないです
            # pred_y1_vector = torch.mm(x_matrix[0].unsqueeze(0), w_matrix)  行列に変換して計算した場合（ただし、softmax は dim=1）
            pred_y1_vector = nn.functional.softmax(pred_y1_vector, dim=0)
            print('y1:')
            print(pred_y1_vector)

            pred_y_vector = torch.mm(x_matrix[:4], w_matrix)
            pred_y_vector = nn.functional.softmax(pred_y_vector, dim=1)
            print('Y:')
            print(pred_y_vector)
        else:
            print('対象ファイルが存在しません。\n')
        print('次の対象ファイルへのパスを入力して下さい。（終了：Enter）')  # 連続して処理を行います
        filepath = input()

if __name__ == '__main__':
    main()

"""
<実行結果>

対象ファイルへのパスを入力して下さい。
train_vectorizer.npz
y1:
tensor([0.1757, 0.4239, 0.2065, 0.1940])
Y:
tensor([[0.1757, 0.4239, 0.2065, 0.1940],
        [0.4047, 0.5529, 0.0213, 0.0211],
        [0.0431, 0.0149, 0.0519, 0.8900],
        [0.0692, 0.8550, 0.0735, 0.0024]])
次の対象ファイルへのパスを入力して下さい。（終了：Enter）
valid_vectorizer.npz
y1:
tensor([0.3654, 0.6025, 0.0150, 0.0172])
Y:
tensor([[0.3654, 0.6025, 0.0150, 0.0172],
        [0.1302, 0.2127, 0.5360, 0.1211],
        [0.0037, 0.9921, 0.0029, 0.0014],
        [0.2681, 0.2388, 0.0796, 0.4135]])
次の対象ファイルへのパスを入力して下さい。（終了：Enter）
test_vectorizer.npz
y1:
tensor([0.1861, 0.3754, 0.0226, 0.4158])
Y:
tensor([[0.1861, 0.3754, 0.0226, 0.4158],
        [0.4023, 0.5048, 0.0599, 0.0330],
        [0.3653, 0.4433, 0.0412, 0.1502],
        [0.6963, 0.1991, 0.0261, 0.0785]])
次の対象ファイルへのパスを入力して下さい。（終了：Enter）


Process finished with exit code 0
"""