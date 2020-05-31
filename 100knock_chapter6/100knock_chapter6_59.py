from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
chapter6_52 = __import__('100knock_chapter6_52')

def main():
    x_train, y_train = chapter6_52.get_dataset('train.feature.txt')
    x_valid, y_valid = chapter6_52.get_dataset('valid.feature.txt')
    x_test, y_test = chapter6_52.get_dataset('test.feature.txt')

    # TfidfVectorizer で単語分割をベクトル化します
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_valid_vec = vectorizer.transform(x_valid)
    x_test_vec = vectorizer.transform(x_test)

    # 正規化パラメータCを変化させます
    c_params = [0.1 * i for i in range(1, 10)]
    c_params += [i for i in range(1, 10)]
    best_score = 0
    # 学習アルゴリズムはロジスティック回帰モデルと線形SVM(サポートベクターマシン)で計算します
    for c in c_params:
        lr = LogisticRegression(solver='liblinear', C=c)
        linear_svc = LinearSVC(C=c)

        lr.fit(x_train_vec, y_train)
        linear_svc.fit(x_train_vec, y_train)

        lr_score =  lr.score(x_valid_vec, y_valid)
        linear_svc_score = linear_svc.score(x_valid_vec, y_valid)

        # ベストスコアと比べて正解率の最も高くなる学習アルゴリズムとパラメータを求めます
        if lr_score > best_score:
            best_c = c
            best_modelname = 'LogisticRegression'
            best_score = lr_score
            best_model = lr
        if linear_svc_score > best_score:
            best_c = c
            best_modelname = 'LinearSVC'
            best_score = linear_svc_score
            best_model = linear_svc

    # パラメータ・学習アルゴリズム・正解率を表示します
    print(best_c)
    print(best_modelname)
    print(best_score)

    # 評価データに用いた時の正解率を表示します
    print(best_model.score(x_test_vec, y_test))


if __name__ == '__main__':
    main()