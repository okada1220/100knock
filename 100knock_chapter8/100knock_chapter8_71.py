import numpy as np
from scipy.special import softmax

def main():
    # ランダムに初期化された行列 W
    np.random.seed(3)
    w_matrix = np.random.rand(300, 4)

    # 学習データの読み込み
    load_array = np.load('train.vectorizer.npz')
    x_train = load_array['arr_0']
    y_train = load_array['arr_1']

    """ y1 を単体で求めると全体で求めた値と丸め込みの差が生じてしまう
    pred_y_train = np.dot(x_train[0], w_matrix)
    print('y1\n', pred_y_train)
    pred_y_train = softmax(pred_y_train)
    print('y1(softmax)\n', pred_y_train)
    """

    # 行列 W からソフトマックス関数で確率を求めます
    pred_y_train = np.dot(x_train[0:4], w_matrix)
    pred_y_train = softmax(pred_y_train, axis=1)

    print('y1(softmax)\n', pred_y_train[0])
    print('Y(softmax)\n', pred_y_train)

    # y1(softmax)
    #  [0.40801148 0.29371007 0.14300672 0.15527174]
    # Y(softmax)
    #  [[0.40801148 0.29371007 0.14300672 0.15527174]
    #  [0.3035311  0.25148309 0.27798297 0.16700283]
    #  [0.39775596 0.26872715 0.15360471 0.17991219]
    #  [0.28088598 0.27626209 0.24483746 0.19801447]]

if __name__ == '__main__':
    main()