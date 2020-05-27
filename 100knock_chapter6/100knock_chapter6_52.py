from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import pickle

# 特徴ファイルからカテゴリと単語分割に分けます
def get_dataset(filename):
    x_data = []
    y_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            category, title = line.split('\t')
            y_data.append(category)
            title = re.sub('\n', '', title)
            x_data.append(title)
    return x_data, y_data

def main():
    x_train, y_train = get_dataset('train.feature.txt')

    # TfidfVectorizer で単語分割をベクトル化します
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)

    # ロジスティック回帰モデルを学習させます
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train_vec, y_train)

    # モデルを保存します
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))

if __name__ == '__main__':
    main()