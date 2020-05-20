import MeCab
import re
import pprint
chapter5_41 = __import__('100knock_chapter5_41')
chapter5_42 = __import__('100knock_chapter5_42')
chapter5_43 = __import__('100knock_chapter5_43')

def main():
    All_Chunk_list = chapter5_41.get_All_Chunk_list('neko.txt.cabocha')
    for j in range(len(All_Chunk_list)):
        for i in range(len(All_Chunk_list[j])):
            chunk_surface = []
            chunk = All_Chunk_list[j][i]
            if chapter5_43.check_chunk(chunk, '名詞'):
                chunk_surface.append(chapter5_42.get_chunk_surface(chunk))

                while(not chunk.dst in chunk.srcs):
                    chunk_surface.append(chapter5_42.get_chunk_surface(All_Chunk_list[j][chunk.dst - 1]))
                    chunk = All_Chunk_list[j][chunk.dst - 1]

                path = ' -> '.join(chunk_surface)
                print(path)

if __name__ == '__main__':
    main()