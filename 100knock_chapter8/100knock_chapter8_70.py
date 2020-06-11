import gensim
import numpy as np
import MeCab

# 問題 50 で構築したデータからベクトル化した行列 x_vector とベクトル y_vector を返します。
def vectorizer_file(filename, action, code):
    # 問題 60 で用いた単語分散表現でベクトル化します
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '../100knock_chapter7/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True)
    # 単語の分割には mecab を使います
    mecab = MeCab.Tagger('-Owakati')
    # flag は array となる x_vector,y_vector の初期化のために使います
    flag = 0
    category_table = {'b':0, 't':1, 'e':2, 'm':3}
    with open(filename, action, encoding=code) as file:
        for line in file:
            category, title = line.split('\t')
            title = title.strip()
            words = mecab.parse(title).split()
            # 単語ベクトルから平均ベクトルを計算します
            words_vector = np.array([model[word].tolist() for word in words if word in model])
            average_vector = np.average(words_vector, axis=0)
            average_vector = np.expand_dims(average_vector, 0)
            category_index = np.array([category_table[category]])
            if flag == 0:
                x_vector = average_vector
                y_vector = category_index
                flag = 1
            else:
                x_vector = np.concatenate([x_vector, average_vector])
                y_vector = np.concatenate([y_vector, category_index])
    return x_vector, y_vector

def main():
    x_train, y_train = vectorizer_file('../100knock_chapter6/train.txt', 'r', 'utf-8')
    x_valid, y_valid = vectorizer_file('../100knock_chapter6/valid.txt', 'r', 'utf-8')
    x_test, y_test = vectorizer_file('../100knock_chapter6/test.txt', 'r', 'utf-8')

    # 各行列・ベクトルは array としてファイルに保存します
    np.savez('train.vectorizer.npz', x_train, y_train)
    np.savez('valid.vectorizer.npz', x_valid, y_valid)
    np.savez('test.vectorizer.npz', x_test, y_test)

if __name__ == '__main__':
    main()