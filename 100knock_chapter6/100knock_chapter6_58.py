from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import japanize_matplotlib
chapter6_52 = __import__('100knock_chapter6_52')

# グラフ化します
def plot_c_params(c_params, train_score, valid_score, test_score):
    plt.plot(c_params, train_score, 'r', label='Train')
    plt.plot(c_params, valid_score, 'b', label='Valid')
    plt.plot(c_params, test_score, 'g', label='Test')
    plt.xlabel('C_params')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

def main():
    x_train, y_train = chapter6_52.get_dataset('train.feature.txt')
    x_valid, y_valid = chapter6_52.get_dataset('valid.feature.txt')
    x_test, y_test = chapter6_52.get_dataset('test.feature.txt')

    # TfidfVectorizer で単語分割をベクトル化します
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_valid_vec = vectorizer.transform(x_valid)
    x_test_vec = vectorizer.transform(x_test)

    # ロジスティック回帰モデルを正規化パラメータを変化させて学習させます
    c_params = [i for i in range(1, 10)]
    train_score = []
    valid_score = []
    test_score = []
    for c in c_params:
        model = LogisticRegression(solver='liblinear', C=c)
        model.fit(x_train_vec, y_train)
        train_score.append(model.score(x_train_vec, y_train))
        valid_score.append(model.score(x_valid_vec, y_valid))
        test_score.append(model.score(x_test_vec, y_test))

    plot_c_params(c_params, train_score, valid_score, test_score)


if __name__ == '__main__':
    main()