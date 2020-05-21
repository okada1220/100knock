import MeCab
import re
import pprint
chapter5_41 = __import__('100knock_chapter5_41')
chapter5_42 = __import__('100knock_chapter5_42')
chapter5_43 = __import__('100knock_chapter5_43')

# chunk に含まれる全ての品詞 pos である morph の基本形を base_list で返す
def celect_morph(chunk, pos):
    base_list = []
    for morph in chunk.morphs:
        if morph.pos == pos:
            base_list.append(morph.base)
    return base_list

# chunk の品詞が pos の語 base_list と chunk に係る助詞 P_base_chunk_list を返す
def search_connection(chunk_list, chunk, pos):
    base_list = celect_morph(chunk, pos)
    P_base_chunk_list = []
    # P_base_chunk_list は一項目に助詞、二項目に助詞を含む chunk となるリスト
    for i in chunk.srcs:
        if not chunk.dst == i:
            P_chunk = chunk_list[i-1]
            P_base_chunk = []
            for P_morph in P_chunk.morphs:
                if P_morph.pos == '助詞':
                    P_base_chunk = [P_morph.base, P_chunk]
            if len(P_base_chunk) > 0:
                P_base_chunk_list.append(P_base_chunk)
    return base_list, P_base_chunk_list

# P_base_chunk_list を転置して P_base と P_chunk ごとのリストに分けて返す
def separate_base_chunk_list(P_base_chunk_list):
    P_base_list = []
    P_chunk_list = []
    for P_base_chunk in P_base_chunk_list:
        P_base_list.append(P_base_chunk[0])
        P_chunk_list.append(P_base_chunk[1])
    return P_base_list, P_chunk_list

def main():
    All_Chunk_list = chapter5_41.get_All_Chunk_list('neko.txt.cabocha')
    for j in range(len(All_Chunk_list)):
        for i in range(len(All_Chunk_list[j])):
            chunk = All_Chunk_list[j][i]

            if chapter5_43.check_chunk(chunk, '動詞'):
                V_base_list, VP_base_chunk_list = search_connection(All_Chunk_list[j], chunk, '動詞')

                VP_base_chunk_list.sort(key=lambda x: x[0])

                VP_base_list, VP_chunk_list = separate_base_chunk_list(VP_base_chunk_list)
                VP_chunk_surface_list = [chapter5_42.get_chunk_surface(VP_chunk) for VP_chunk in VP_chunk_list]  # Chunk を表層格にする
                VP_base = ' '.join(VP_base_list)
                VP_chunk_surface = ' '.join(VP_chunk_surface_list)

                print(V_base_list[0] + '\t' + VP_base + '\t' + VP_chunk_surface)

if __name__ == '__main__':
    main()