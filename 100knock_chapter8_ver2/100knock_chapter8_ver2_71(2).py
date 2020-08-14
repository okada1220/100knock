import os
import torch
import utils
import models

def main():
    print('対象ファイルへのパスを入力して下さい。')
    filepath = input()

    while(filepath != ''):
        if os.path.exists(filepath):
            x_matrix, y_vector = utils.load_vectorizer_file(filepath)
            x_matrix = torch.from_numpy(x_matrix)
            y_vector = torch.from_numpy(y_vector)

            INPUT_SIZE = 300
            OUTPUT_SIZE = 4
            singlenet_model = models.SingleNet(INPUT_SIZE, OUTPUT_SIZE)

            pred_y1_vector = singlenet_model.forward(x_matrix[:1])
            print('y1:')
            print(pred_y1_vector)

            pred_y_matrix = singlenet_model.forward(x_matrix[:4])
            print('Y:')
            print(pred_y_matrix)
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
tensor([0.2447, 0.2583, 0.2396, 0.2574], grad_fn=<SoftmaxBackward>)
Y:
tensor([[0.2447, 0.2583, 0.2396, 0.2574],
        [0.2516, 0.2692, 0.2260, 0.2533],
        [0.2455, 0.2542, 0.2414, 0.2589],
        [0.2588, 0.2711, 0.2303, 0.2399]], grad_fn=<CatBackward>)
次の対象ファイルへのパスを入力して下さい。（終了：Enter）
valid_vectorizer.npz
y1:
tensor([0.2324, 0.2614, 0.2593, 0.2470], grad_fn=<SoftmaxBackward>)
Y:
tensor([[0.2324, 0.2614, 0.2593, 0.2470],
        [0.2463, 0.2536, 0.2447, 0.2554],
        [0.2308, 0.2537, 0.2473, 0.2683],
        [0.2350, 0.2623, 0.2536, 0.2490]], grad_fn=<CatBackward>)
次の対象ファイルへのパスを入力して下さい。（終了：Enter）
test_vectorizer.npz
y1:
tensor([0.2402, 0.2487, 0.2464, 0.2647], grad_fn=<SoftmaxBackward>)
Y:
tensor([[0.2402, 0.2487, 0.2464, 0.2647],
        [0.2508, 0.2565, 0.2436, 0.2490],
        [0.2417, 0.2561, 0.2605, 0.2417],
        [0.2503, 0.2560, 0.2369, 0.2568]], grad_fn=<CatBackward>)
次の対象ファイルへのパスを入力して下さい。（終了：Enter）


Process finished with exit code 0
"""