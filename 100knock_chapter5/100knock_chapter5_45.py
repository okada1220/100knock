import MeCab
import re
import pprint
chapter5_41 = __import__('100knock_chapter5_41')
chapter5_42 = __import__('100knock_chapter5_42')
chapter5_43 = __import__('100knock_chapter5_43')

def search_pattern(chunk_list, V_chunk):
    V_base = ''
    P_base_list = []
    for V_morph in V_chunk.morphs:
        if V_morph.pos == '動詞':
            V_base = V_morph.base
            break
    for i in V_chunk.srcs:
        if not V_chunk.dst == i:
            P_chunk = chunk_list[i-1]
            P_base = ''
            for P_morph in P_chunk.morphs:
                if P_morph.pos == '助詞':
                    P_base = P_morph.base
            P_base_list.append(P_base)
    return V_base, P_base_list

def main():
    All_Chunk_list = chapter5_41.get_All_Chunk_list('neko.txt.cabocha')
    for j in range(len(All_Chunk_list)):
        for i in range(len(All_Chunk_list[j])):
            chunk = All_Chunk_list[j][i]
            if chapter5_43.check_chunk(chunk, '動詞'):
                V_base, P_base_list = search_pattern(All_Chunk_list[j], chunk)
                P_base_list.sort()
                P_base = ' '.join(P_base_list)
                print(V_base + '\t' + P_base)

if __name__ == '__main__':
    main()