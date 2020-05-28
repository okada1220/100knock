from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pickle
chapter6_52 = __import__('100knock_chapter6_52')

def main():
    # ファイルからカテゴリと単語分割に分けます
    x_train, y_train = chapter6_52.get_dataset('train.feature.txt')
    x_test, y_test = chapter6_52.get_dataset('test.feature.txt')

    # ベクトル化します
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # 学習したモデルから混合行列を計測します
    with open('model.sav', 'rb') as f:
        load_model = pickle.load(f)
    y_train_pred = load_model.predict(x_train_vec)
    y_test_pred = load_model.predict(x_test_vec)
    print(confusion_matrix(y_train, y_train_pred, labels=['b', 't', 'e', 'm']))
    print(confusion_matrix(y_test, y_test_pred, labels=['b', 't', 'e', 'm']))

if __name__ == '__main__':
    main()