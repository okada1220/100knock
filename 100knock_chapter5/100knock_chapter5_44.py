import MeCab
import CaboCha
import re
import pprint
import pydotplus
chapter5_40 = __import__('100knock_chapter5_40')
chapter5_41 = __import__('100knock_chapter5_41')
chapter5_42 = __import__('100knock_chapter5_42')

def main():
    print('文を入力してください。')
    original_text = input()
    cabocha = CaboCha.Parser()
    cabocha_text = cabocha.parseToString(original_text)
    cabocha_line = cabocha_text.split('\n')

    mecab = MeCab.Tagger()
    Chunk_list = []
    item = 0
    dst_compile = re.compile('\-+D')
    for line in cabocha_line:
        if not line == 'EOS':
            Morph_list = []
            item += 1
            line = line.strip()
            dst_search = dst_compile.search(line)
            dst = int((line.count('-') + 1) / 2) + item
            line = re.sub(r'\-+D|\s|\|', '', line)
            word = mecab.parse(line).split('\n')
            word.remove('')
            word.remove('EOS')
            for morph_info in word:
                morph_list = re.split(r'\t|,', morph_info)
                Morph_list += [chapter5_40.Morph(morph_list)]
            Chunk_list += [chapter5_41.Chunk(Morph_list, dst)]

    count = 0
    for chunk in Chunk_list:
        count += 1
        Chunk_list[chunk.dst - 1].set_srcs(count + 1)

    text = 'digraph {\n'
    text += 'node [fontname = "MS Gothic"];\n'
    for chunk in Chunk_list:
        chunk_surface = chapter5_42.get_chunk_surface(chunk)
        if chunk_surface == '':
            continue
        if not chunk.dst in chunk.srcs:
            target_chunk = Chunk_list[chunk.dst - 1]
            target_chunk_surface = chapter5_42.get_chunk_surface(target_chunk)
            text += "\"" + target_chunk_surface + '\" -> \"' + chunk_surface + '\"\n'
    text += '}\n'

    graph = pydotplus.graph_from_dot_data(text)
    graph.write_png('iris3.png')

if __name__ == '__main__':
    main()