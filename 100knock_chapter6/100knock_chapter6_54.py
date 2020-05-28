from sklearn.feature_extraction.text import TfidfVectorizer
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

    # 学習したモデルから正解率を計測します
    with open('model.sav', 'rb') as f:
        load_model = pickle.load(f)
    study = load_model.score(x_train_vec, y_train)
    print(study)
    result = load_model.score(x_test_vec, y_test)
    print(result)

if __name__ == '__main__':
    main()