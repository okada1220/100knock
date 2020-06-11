import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score

def main():
    # 行列 W を読み込みます
    w_matrix = np.load('w_matrix.npy')
    print(w_matrix)

    # 学習データの読み込み
    load_array = np.load('train.vectorizer.npz')
    x_train = load_array['arr_0']
    y_train = load_array['arr_1']

    load_array = np.load('test.vectorizer.npz')
    x_test = load_array['arr_0']
    y_test = load_array['arr_1']

    # 全ての事例 x_train を行列 W からソフトマックス関数で確率・予測を求めます
    pred_y_train = np.dot(x_train, w_matrix)
    pred_y_train = softmax(pred_y_train, axis=1)
    pred_y_train = np.argmax(pred_y_train, axis=1)

    # 全ての事例 x_test を行列 W からソフトマックス関数で確率・予測を求めます
    pred_y_test = np.dot(x_test, w_matrix)
    pred_y_test = softmax(pred_y_test, axis=1)
    pred_y_test = np.argmax(pred_y_test, axis=1)

    # それぞれの正答率を求めます
    print('accuracy(train):', accuracy_score(y_train, pred_y_train))
    print('accuracy(test):', accuracy_score(y_test, pred_y_test))

if __name__ == '__main__':
    main()