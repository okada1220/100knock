import MeCab
import re
import pprint

class Morph:
    def __init__(self, morph_list):
        self.surface = morph_list[0]
        self.base = morph_list[7]
        self.pos = morph_list[1]
        self.pos1 = morph_list[2]

def main():
    mecab = MeCab.Tagger()
    All_Morph_list = []
    Morph_list = []
    with open('neko.txt.cabocha', 'r', encoding='utf-8') as file:
        for line in file:
            if not line == 'EOS\n':
                line = line.strip()
                line = re.sub(r'[-+D]|\s|\|', '', line)
                word = mecab.parse(line).split('\n')
                word.remove('')
                word.remove('EOS')
                for morph_info in word:
                    morph_list = re.split(r'\t|,', morph_info)
                    Morph_list += [Morph(morph_list)]
            else:
                if len(Morph_list) > 0:
                    All_Morph_list.append(Morph_list)
                    Morph_list = []

    pprint.pprint(All_Morph_list[2])
    for i in range(len(All_Morph_list[2])):
        x = All_Morph_list[2][i]
        print(x.surface, x.base, x.pos, x.pos1)

if __name__ == '__main__':
    main()