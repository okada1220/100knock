import numpy as np
from scipy.special import softmax

def main():
    # ランダムに初期化された行列 W （同じランダムであるよう seed を用います）
    np.random.seed(3)
    w_matrix = np.random.rand(300, 4)

    # 学習データの読み込み + ラベルベクトルを０，１の行列 y_train_array に変換します
    load_array = np.load('train.vectorizer.npz')
    x_train = load_array['arr_0']
    y_train = load_array['arr_1']
    y_train_list = [[i == train for i in range(4)] for train in y_train]
    y_train_array = np.array(y_train_list)

    # 全ての事例 x_train を行列 W からソフトマックス関数で確率を求めます
    pred_y_train = np.dot(x_train, w_matrix)
    pred_y_train = softmax(pred_y_train, axis=1)

    # 勾配降下率を parameter 、エポック回数を epoch とします
    parameter = 0.01
    epoch = 10
    for x in range(epoch):
        for index, train in enumerate(x_train):
            # 事例 train のクロスエントロピー損失 loss と勾配 grad_loss を求めます
            loss = -np.log(pred_y_train[index][y_train[index]])
            grad_loss = [[(pred_y_train[index][i] - y_train_array[index][i]) * train[j] for i in range(4)] for j in range(300)]
            grad_loss = np.array(grad_loss)
            print(loss)
            # 行列 W を更新し、それによるソフトマックス関数も更新します
            w_matrix -= parameter * grad_loss
            pred_y_train = np.dot(x_train, w_matrix)
            pred_y_train = softmax(pred_y_train, axis=1)

    # 最終的に求まった行列 W を array で保存します
    np.save('w_matrix.npy', w_matrix)

if __name__ == '__main__':
    main()