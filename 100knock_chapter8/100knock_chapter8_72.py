import numpy as np
from scipy.special import softmax
import pprint

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

    # 行列 W からソフトマックス関数で確率を求めます
    pred_y_train = np.dot(x_train[0:4], w_matrix)
    pred_y_train = softmax(pred_y_train, axis=1)

    # 事例 x1 のクロスエントロピー損失 loss と勾配 grad_loss を求めます
    loss = -np.log(pred_y_train[0][y_train[0]])
    grad_loss = [[(pred_y_train[0][i] - y_train_array[0][i]) * x_train[0][j] for i in range(4)] for j in range(300)]
    grad_loss = np.array(grad_loss)
    print('y1(loss)\n', loss)
    print('y1(grad)')
    pprint.pprint(grad_loss)

    # 事例集合 x1,x2,x3,x4 のクロスエントロピー損失 grouploss と勾配 grad_grouploss を求めます
    grouploss_list = [-np.log(pred_y_train[i][y_train[i]]) for i in range(4)]
    grouploss = sum(grouploss_list) / len(grouploss_list)
    print('Y(loss)\n', grouploss)
    grad_grouploss = [[[(pred_y_train[k][i] - y_train_array[k][i]) * x_train[k][j] for i in range(4)] for j in range(300)] for k in range(4)]
    grad_grouploss = np.array(grad_grouploss)
    grad_grouploss = np.average(grad_grouploss, axis=0)
    print('Y(grad)')
    pprint.pprint(grad_grouploss)

    # y1(loss)
    #  1.944863681756017
    # y1(grad)
    # array([[ 0.02687265,  0.01934447, -0.0564437 ,  0.01022658],
    #        [ 0.01606133,  0.01156187, -0.03373546,  0.00611226],
    #        [-0.01654468, -0.01190981,  0.03475068, -0.0062962 ],
    #        ...,
    #        [-0.03119219, -0.02245393,  0.06551654, -0.01187042],
    #        [-0.00317853, -0.00228809,  0.00667624, -0.00120961],
    #        [ 0.00893793,  0.00643403, -0.01877336,  0.00340139]])
    # Y(loss)
    #  1.570078529226092
    # Y(grad)
    # array([[ 0.03340979, -0.00404236, -0.02573415, -0.00363328],
    #        [ 0.01389847,  0.00022368, -0.0139049 , -0.00021725],
    #        [ 0.0188752 , -0.01047074, -0.00166434, -0.00674012],
    #        ...,
    #        [-0.04792319,  0.005951  ,  0.03699562,  0.00497657],
    #        [-0.01521569,  0.00873177,  0.00034574,  0.00613818],
    #        [-0.00432635,  0.00705756, -0.0072592 ,  0.00452799]])

if __name__ == '__main__':
    main()