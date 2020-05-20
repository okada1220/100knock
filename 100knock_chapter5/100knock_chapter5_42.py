import MeCab
import re
import pprint
chapter5_41 = __import__('100knock_chapter5_41')

def get_chunk_surface(chunk):
    chunk_surface = ''
    for morph in chunk.morphs:
        if not morph.pos == '記号':
            chunk_surface += morph.surface
    return chunk_surface

def main():
    All_Chunk_list = chapter5_41.get_All_Chunk_list('neko.txt.cabocha')
    for j in range(len(All_Chunk_list)):
        for i in range(len(All_Chunk_list[j])):
            chunk = All_Chunk_list[j][i]
            chunk_surface = get_chunk_surface(chunk)
            if not (chunk.dst - 1) == i:
                target_chunk = All_Chunk_list[j][chunk.dst - 1]
                target_chunk_surface = get_chunk_surface(target_chunk)
                print(chunk_surface + '\t' + target_chunk_surface)

if __name__ == '__main__':
    main()