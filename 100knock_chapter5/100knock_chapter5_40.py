import MeCab
import re
import pprint

class Morph:
    def __init__(self, morph_list):
        self.surface = morph_list[0]  # 表層格
        self.base = morph_list[7]  # 基本形
        self.pos = morph_list[1]  # 品詞
        self.pos1 = morph_list[2] # 品詞細分類１

def main():
    mecab = MeCab.Tagger()
    All_Morph_list = []  # Morph_list を全文まとめたリスト
    Morph_list = []  # 各文の Morph をリスト化
    with open('neko.txt.cabocha', 'r', encoding='utf-8') as file:
        for line in file:
            # EOS で文の区切りを判断
            if not line == 'EOS\n':
                # 形態素に分割
                line = line.strip()
                line = re.sub(r'\-+D|\s|\|', '', line)
                word = mecab.parse(line).split('\n')
                word.remove('')
                word.remove('EOS')
                for morph_info in word:
                    morph_list = re.split(r'\t|,', morph_info)
                    Morph_list += [Morph(morph_list)]
            else:
                if len(Morph_list) > 0:  # EOSが二行連続で出る場合を除去
                    All_Morph_list.append(Morph_list)
                    Morph_list = []

    pprint.pprint(All_Morph_list[2])
    for morph in All_Morph_list[2]:
        print(morph.surface, morph.base, morph.pos, morph.pos1)

if __name__ == '__main__':
    main()