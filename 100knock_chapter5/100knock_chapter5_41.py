import MeCab
import re
import pprint
chapter5_40 = __import__('100knock_chapter5_40')

class Chunk:
    def __init__(self, morphs, dst):
        self.morphs = morphs
        self.dst = dst
        self.srcs = []
    def set_srcs(self, srcs):  # Chunk の srcs はファイルの走査後に決定
        self.srcs.append(srcs)

# 全文を文ごとに分けて Chunk ごとに分けた二次元リスト All_Chunk_list を返す
def get_All_Chunk_list(filename):
    mecab = MeCab.Tagger()
    All_Chunk_list = []
    Chunk_list = []
    item = 0  # 文中の item 番目の Chunk とするため
    dst_compile = re.compile('\-+D')  # dst の判断用
    with open('neko.txt.cabocha', 'r', encoding='utf-8') as file:
        for line in file:
            if not line == 'EOS\n':
                Morph_list = []  # Chunk 中の morphs
                item += 1
                line = line.strip()
                # - の数と D から dst を計算
                dst_search = dst_compile.search(line)
                dst = int((line.count('-') + 1) / 2) + item

                line = re.sub(r'\-+D|\s|\|', '', line)
                word = mecab.parse(line).split('\n')
                word.remove('')
                word.remove('EOS')
                for morph_info in word:
                    morph_list = re.split(r'\t|,', morph_info)
                    Morph_list += [chapter5_40.Morph(morph_list)]
                Chunk_list += [Chunk(Morph_list, dst)]
            else:
                if len(Chunk_list) > 0:  # EOS が二連続で出る場合を除去
                    All_Chunk_list.append(Chunk_list)
                    Chunk_list = []
                    item = 0

    # dst から srcs を決定
    for x in range(len(All_Chunk_list)):
        for y in range(len(All_Chunk_list[x])):
            target = All_Chunk_list[x][y]
            All_Chunk_list[x][target.dst - 1].set_srcs(y + 1)

    return All_Chunk_list

def main():
    All_Chunk_list = get_All_Chunk_list('neko.txt.cabocha')
    pprint.pprint(All_Chunk_list[7])
    for chunk in All_Chunk_list[7]:
        for morph in chunk.morphs:
            pprint.pprint(morph.surface)
        print('→ 係り先は' + str(chunk.dst) + '項目です。')
        for src in chunk.srcs:
            print('← 係り元は' + str(src) + '項目です。')

if __name__ == '__main__':
    main()