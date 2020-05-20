import MeCab
import re
import pprint
chapter5_41 = __import__('100knock_chapter5_41')
chapter5_42 = __import__('100knock_chapter5_42')

def check_chunk(chunk, pos):
    pos_flag = 0
    for morph in chunk.morphs:
        if morph.pos == pos:
            pos_flag = 1
    return pos_flag

def main():
    All_Chunk_list = chapter5_41.get_All_Chunk_list('neko.txt.cabocha')
    for j in range(len(All_Chunk_list)):
        for i in range(len(All_Chunk_list[j])):
            chunk = All_Chunk_list[j][i]
            N_flag = check_chunk(chunk, '名詞')
            chunk_surface = chapter5_42.get_chunk_surface(chunk)
            if not (chunk.dst - 1) == i:
                target_chunk = All_Chunk_list[j][chunk.dst - 1]
                V_flag = check_chunk(target_chunk, '動詞')
                if N_flag == 1 and V_flag == 1:
                    target_chunk_surface = chapter5_42.get_chunk_surface(target_chunk)
                    print(chunk_surface + '\t' + target_chunk_surface)

if __name__ == '__main__':
    main()