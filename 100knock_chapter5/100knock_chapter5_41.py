import MeCab
import re
import pprint
chapter5_40 = __import__('100knock_chapter5_40')

class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []
    def set_srcs(self, srcs):
        self.srcs.append(srcs)

def main():
    mecab = MeCab.Tagger()
    All_Chunk_list = []
    Chunk_list = []
    item = 0
    dst_compile = re.compile('[-+D]')
    with open('neko.txt.cabocha', 'r', encoding='utf-8') as file:
        for line in file:
            if not line == 'EOS\n':
                Morph_list = []
                item +=1
                line.strip()
                dst_search = dst_compile.search(line)
                dst = int((line.count('-') + 1)/ 2) + item
                line = re.sub(r'[-+D]|\s|\|', '', line)
                word = mecab.parse(line).split('\n')
                word.remove('')
                word.remove('EOS')
                for morph_info in word:
                    morph_list = re.split(r'\t|,', morph_info)
                    Morph_list += [chapter5_40.Morph(morph_list)]
                Chunk_list += [Chunk(Morph_list, dst)]
            else:
                if len(Chunk_list) > 0:
                    All_Chunk_list.append(Chunk_list)
                    Chunk_list = []
                    item = 0

    for x in range(len(All_Chunk_list)):
        for y in range(len(All_Chunk_list[x])):
            target = All_Chunk_list[x][y]
            All_Chunk_list[x][target.dst - 1].set_srcs(y + 1)

    pprint.pprint(All_Chunk_list[7])
    for i in range(len(All_Chunk_list[7])):
        z = All_Chunk_list[7][i]
        for morph in z.morphs:
            pprint.pprint(morph.surface)
        print('→ 係り先は' + str(z.dst) + '項目です。')
        for src in z.srcs:
            print('← 係り元は' + str(src) + '項目です。')

if __name__ == '__main__':
    main()