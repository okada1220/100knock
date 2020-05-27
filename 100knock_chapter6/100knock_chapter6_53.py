from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import MeCab
chapter6_52 = __import__('100knock_chapter6_52')

def main():
    # 入力された記事見出しから予測結果、予測確率を計算します
    print('記事見出しを入力してください。')
    title = input()

    # 記事見出しを単語分割をします
    mecab = MeCab.Tagger("-Owakati")
    words = mecab.parse(title)
    words = re.sub(' \n', '', words)

    # 記事見出しをベクトル化します
    x_train, y_train = chapter6_52.get_dataset('train.feature.txt')
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    words_vec = vectorizer.transform([words])

    # 学習したモデルを用いて予測を行います
    load_model = pickle.load(open('model.sav', 'rb'))
    y_pred = load_model.predict(words_vec)
    y_pred_proba = load_model.predict_proba(words_vec)
    print(y_pred)
    print(y_pred_proba)

if __name__ == '__main__':
    main()