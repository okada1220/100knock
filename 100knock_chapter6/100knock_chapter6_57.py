from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import heapq
chapter6_52 = __import__('100knock_chapter6_52')

# 重みを高い順にトップ１０を largest、低い順にトップ１０を smallestとして対応する単語を namesから得る
def get_feature_word(number, load_model, names):
    weight = load_model.coef_[number]
    weight = weight.tolist()
    largest = heapq.nlargest(10, weight)
    smallest = heapq.nsmallest(10, weight)
    largest_names_list = [names[weight.index(largest[i])] for i in range(10)]
    smallest_names_list = [names[weight.index(smallest[i])] for i in range(10)]
    print(number, 'largest', '\t'.join(largest_names_list))
    print(number, 'smallest', '\t'.join(smallest_names_list))

def main():
    # ファイルからカテゴリと単語分割に分けます
    x_train, y_train = chapter6_52.get_dataset('train.feature.txt')

    # TfidfVectorizer に fit させた単語を names として得ます
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(x_train)
    names = vectorizer.get_feature_names()

    # モデルから重みを得ます
    load_model = pickle.load(open('model.sav', 'rb'))
    # クラス分類に用いる線が４本ある　-> ４つの重みのパターンがある
    for x in range(4):
        get_feature_word(x, load_model, names)

if __name__ == '__main__':
    main()