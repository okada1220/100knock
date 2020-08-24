import os
import gensim
import torch
import utils

def main():
    # モデル、カテゴリ-ラベル対を用意します
    model = gensim.models.KeyedVectors.load_word2vec_format(
        '../100knock_chapter7/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', binary=True
    )
    category_table = {'b':0, 't':1, 'e':2, 'm':3}

    # ファイル名を入力
    print('対象ファイルへのパスを入力して下さい。')
    filepath = input()
    while(filepath != ''):  # 連続処理
        if os.path.exists(filepath):  # 例外処理
            # 特徴行列Ｘと正解ベクトルＹを得ます
            x_matrix, y_vector = utils.make_vector(filepath, model, category_table)

            # 出力ファイル名を入力ファイル名から決めます
            file = filepath.split('/')[-1]
            filename = file.split('.')
            make_filename = '.'.join(filename[:-1]) + '_vectorizer.pth'
            # 保存します
            torch.save({'x_matrix':x_matrix,
                        'y_vector':y_vector},
                       make_filename)
            print(make_filename + 'を作成しました。\n')
        else:
            print('対象ファイルが存在しません。\n')
        print('次の対象ファイルへのパスを入力して下さい。（終了：Enter）')  # 連続して処理を行います
        filepath = input()

if __name__ == '__main__':
    main()

"""
<実行結果>

対象ファイルへのパスを入力して下さい。
../100knock_chapter6/train.txt
train_vectorizer.pthを作成しました。

次の対象ファイルへのパスを入力して下さい。（終了：Enter）
../100knock_chapter6/valid.txt
valid_vectorizer.pthを作成しました。

次の対象ファイルへのパスを入力して下さい。（終了：Enter）
../100knock_chapter6/test.txt
test_vectorizer.pthを作成しました。

次の対象ファイルへのパスを入力して下さい。（終了：Enter）
../100knock_chapter6/hogehoge.txt
対象ファイルが存在しません。

次の対象ファイルへのパスを入力して下さい。（終了：Enter）


Process finished with exit code 0
"""