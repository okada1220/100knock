import MeCab
import re
import pprint
chapter5_41 = __import__('100knock_chapter5_41')
chapter5_42 = __import__('100knock_chapter5_42')
chapter5_43 = __import__('100knock_chapter5_43')
chapter5_46 = __import__('100knock_chapter5_46')

def main():
    All_Chunk_list = chapter5_41.get_All_Chunk_list('neko.txt.cabocha')
    for j in range(len(All_Chunk_list)):
        for i in range(len(All_Chunk_list[j])):
            chunk = All_Chunk_list[j][i]

            if chapter5_43.check_chunk(chunk, '動詞'):
                V_base_list, VP_base_chunk_list = chapter5_46.search_connection(All_Chunk_list[j], chunk, '動詞')
                VP_base_chunk_list.sort(key=lambda x: x[0])

                # 「サ行変格接続名詞＋を」が係っていれば、動詞とまとめて multiV_base とする
                for VP_base_chunk in VP_base_chunk_list:
                    VP_chunk = VP_base_chunk[1]
                    if VP_chunk.morphs[0].pos1 == 'サ変接続' and VP_chunk.morphs[1].base == 'を' and len(VP_chunk.morphs) == 2:
                        multiV_base = chapter5_42.get_chunk_surface(VP_chunk) + V_base_list[0]
                        VP_base_chunk_list.remove(VP_base_chunk)

                        VP_base_list, VP_chunk_list = chapter5_46.separate_base_chunk_list(VP_base_chunk_list)
                        VP_chunk_surface_list = [chapter5_42.get_chunk_surface(VP_chunk) for VP_chunk in VP_chunk_list]
                        VP_base = ' '.join(VP_base_list)
                        VP_chunk_surface = ' '.join(VP_chunk_surface_list)

                        print(multiV_base + '\t' + VP_base + '\t' + VP_chunk_surface)

if __name__ == '__main__':
    main()