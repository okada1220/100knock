# -*- coding: utf-8 -*-

import sys
import os
import utils

def main():
    if len(sys.argv) < 2:
        print('python 100knock_chapter9_ver2_80 [dict_filepath]')
        sys.exit()
    filepath = sys.argv[1]
    while(1):
        if os.path.exists(filepath):
            word_id_dict = utils.make_word_id_dict(filepath)
            break
        else:
            print('対象の辞書となるファイルが存在しません。')
            print('もう一度、入力して下さい。')
            filepath = input()

    print('ID化したい文を入力して下さい。')
    sentence = input()
    while(sentence != ''):
        words_id = utils.trans_words_id(word_id_dict, sentence)
        print(words_id, '\n')
        print('次の文を入力して下さい。（終了：Enter）')
        sentence = input()
if __name__ == '__main__':
    main()

"""
<実行結果>

ID化したい文を入力して下さい。
Lena Dunham: I Feel Prettier With A Naked Face And ChapStick
tensor([ 753,  754,    8,   92, 5114,    0,   27,   28, 2915,  645,   30,    0])

次の文を入力して下さい。（終了：Enter）
REFILE-FOREX-Yen grinds lower as global stocks rally, dollar holds steady
tensor([  0,   0, 198,  11, 533, 190, 589,   2, 295, 677, 493])

次の文を入力して下さい。（終了：Enter）

"""