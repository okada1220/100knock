import MeCab
import re
import pprint
chapter5_41 = __import__('100knock_chapter5_41')
chapter5_42 = __import__('100knock_chapter5_42')
chapter5_43 = __import__('100knock_chapter5_43')

def N_index(chunk, index):
    index_surface = ''
    for morph in chunk.morphs:
        if morph.pos == '名詞':
            if not index_surface == index:
                index_surface += index
        else:
            index_surface += morph.surface
    return index_surface

def main():
    All_Chunk_list = chapter5_41.get_All_Chunk_list('neko.txt.cabocha')
    All_chunk_surface = []
    for j in range(len(All_Chunk_list)):
        chunk_list_line = []
        for i in range(len(All_Chunk_list[j])):
            chunk_list = []
            chunk = All_Chunk_list[j][i]
            if chapter5_43.check_chunk(chunk, '名詞'):
                chunk_list.append(chunk)

                while(not chunk.dst in chunk.srcs):
                    chunk_list.append(All_Chunk_list[j][chunk.dst - 1])
                    chunk = All_Chunk_list[j][chunk.dst - 1]
                chunk_list_line.append(chunk_list)

        if len(chunk_list_line) > 1:
            for X_path_number in range(len(chunk_list_line)):
                for Y_path_number in range(X_path_number + 1, len(chunk_list_line)):
                    chunk_surface = ''
                    flag = 0
                    for X_path_chunk_number in range(len(chunk_list_line[X_path_number])):
                        X_path_chunk = chunk_list_line[X_path_number][X_path_chunk_number]
                        if X_path_chunk_number == 0:
                            chunk_surface = N_index(X_path_chunk, 'X')
                        elif X_path_chunk == chunk_list_line[Y_path_number][0]:
                            chunk_surface += ' -> Y'
                            break
                        elif X_path_chunk in chunk_list_line[Y_path_number] and flag == 0:
                            chunk_surface += ' | '
                            for Y_path_chunk_number in range(len(chunk_list_line[Y_path_number])):
                                Y_path_chunk = chunk_list_line[Y_path_number][Y_path_chunk_number]
                                if Y_path_chunk_number == 0:
                                    chunk_surface += N_index(Y_path_chunk, 'Y')
                                elif Y_path_chunk == X_path_chunk:
                                    chunk_surface += ' | '
                                    flag = 1
                                    break
                                else:
                                    chunk_surface += ' -> ' + chapter5_42.get_chunk_surface(Y_path_chunk)
                            chunk_surface += chapter5_42.get_chunk_surface(X_path_chunk)
                        else:
                            chunk_surface += ' -> ' + chapter5_42.get_chunk_surface(X_path_chunk)
                    print(chunk_surface)

if __name__ == '__main__':
    main()