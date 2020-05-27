from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
import pickle
chapter6_52 = __import__('100knock_chapter6_52')

# カテゴリが index となるものの適合率・再現率・Ｆ１を求めて返します
def get_score(index, test, pred):
    index_test = []
    index_pred = []
    for i in test:
        if i == index:
            index_test.append(1)
        else:
            index_test.append(0)
    for j in pred:
        if j == index:
            index_pred.append(1)
        else:
            index_pred.append(0)
    precision = precision_score(index_test, index_pred)
    recall = recall_score(index_test, index_pred)
    f1 = f1_score(index_test, index_pred)
    print(index, precision, recall, f1)
    return precision, recall, f1

def main():
    # ファイルからカテゴリと単語分割に分けます
    x_train, y_train = chapter6_52.get_dataset('train.feature.txt')
    x_test, y_test = chapter6_52.get_dataset('test.feature.txt')

    # ベクトル化します
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # 学習したモデルからそれぞれの評価を計測します
    load_model = pickle.load(open('model.sav', 'rb'))
    y_pred = load_model.predict(x_test_vec)
    # 各カテゴリごとの適合率・再現率・Ｆ１スコアを求めます
    b_precision, b_recall, b_f1 = get_score('b', y_test, y_pred)
    t_precision, t_recall, t_f1 = get_score('t', y_test, y_pred)
    e_precision, e_recall, e_f1 = get_score('e', y_test, y_pred)
    m_precision, m_recall, m_f1 = get_score('m', y_test, y_pred)
    # マイクロ平均を求めます
    count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            count += 1
    micro_precision = count / len(y_pred)
    micro_recall = count / len(y_test)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    print('micro', micro_precision, micro_recall, micro_f1)
    # マクロ平均を求めます
    macro_precision = (b_precision + t_precision + e_precision + m_precision) / 4
    macro_recall = (b_recall + t_recall + e_recall + m_recall) / 4
    # Ｆ１スコアについては二つの定義でマクロ平均を求めます
    macro_f1_1 = (b_f1 + t_f1 + e_f1 + m_f1) / 4  # カテゴリの平均から
    macro_f1_2 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)  # 適合率・再現率のマクロ平均から
    print('macro', macro_precision, macro_recall, macro_f1_1, macro_f1_2)

if __name__ == '__main__':
    main()