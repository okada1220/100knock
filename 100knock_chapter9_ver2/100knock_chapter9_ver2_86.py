# -*- coding: utf-8 -*-

import sys
import os
import utils
import models

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

    input_size = max(word_id_dict.values()) + 1
    embed_size = 300
    hidden_size = 50
    output_size = 4
    model = models.CNN(input_size, embed_size, hidden_size, output_size)

    category_table = {'b':0, 't':1, 'e':2, 'm':3}
    print('予測したい文を入力して下さい。')
    sentence = input()
    while (sentence != ''):
        words_id = utils.trans_words_id(word_id_dict, sentence)
        output = model.forward(words_id.unsqueeze(0))
        print("output", output)
        output = output.argmax().item()
        for key, value in category_table.items():
            if value == output:
                output = key
                break
        print("output", output, '\n')
        print('次の文を入力して下さい。（終了：Enter）')
        sentence = input()
if __name__ == '__main__':
    main()

"""
<実行結果>

予測したい文を入力して下さい。
Lena Dunham: I Feel Prettier With A Naked Face And ChapStick
output tensor([0.3609, 0.1582, 0.2228, 0.2581], grad_fn=<SoftmaxBackward>)
output b

次の文を入力して下さい。（終了：Enter）
REFILE-FOREX-Yen grinds lower as global stocks rally, dollar holds steady
output tensor([0.2956, 0.1853, 0.2052, 0.3139], grad_fn=<SoftmaxBackward>)
output m

次の文を入力して下さい。（終了：Enter）
"""