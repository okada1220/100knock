import MeCab
import re
import pprint
chapter5_41 = __import__('100knock_chapter5_41')
chapter5_42 = __import__('100knock_chapter5_42')
chapter5_43 = __import__('100knock_chapter5_43')

# chunk 中の名詞を　index で置き換えた表層格 index_surface を返す
def N_index(chunk, index):
    index_surface = ''
    for morph in chunk.morphs:
        if morph.pos == '名詞':
            if not index_surface == index:  # 名詞が連続するものを一つの index とする
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

                # 一文ごとのパスを chunk_list_line をしてリストにする
                while(not chunk.dst in chunk.srcs):
                    chunk_list.append(All_Chunk_list[j][chunk.dst - 1])
                    chunk = All_Chunk_list[j][chunk.dst - 1]
                chunk_list_line.append(chunk_list)

        if len(chunk_list_line) > 1:
            # 一つ目のパスを X_path とする
            for X_path_number in range(len(chunk_list_line)):
                # 二つ目のパスを Y_path とする
                for Y_path_number in range(X_path_number + 1, len(chunk_list_line)):
                    chunk_surface = ''
                    flag = 0  # 初めて X_path と Y_path が被る瞬間だけ Y_path の表示に移行するため
                    for X_path_chunk_number in range(len(chunk_list_line[X_path_number])):
                        X_path_chunk = chunk_list_line[X_path_number][X_path_chunk_number]
                        # X_path の一つ目の Chunk をindex化して表示
                        if X_path_chunk_number == 0:
                            chunk_surface = N_index(X_path_chunk, 'X')
                        # Y_path のスタートにぶつかったら -> Y で終了
                        elif X_path_chunk == chunk_list_line[Y_path_number][0]:
                            chunk_surface += ' -> Y'
                            break
                        # Y_path の途中にぶつかったら | として Y_path を表示する
                        elif X_path_chunk in chunk_list_line[Y_path_number] and flag == 0:
                            chunk_surface += ' | '
                            for Y_path_chunk_number in range(len(chunk_list_line[Y_path_number])):
                                Y_path_chunk = chunk_list_line[Y_path_number][Y_path_chunk_number]
                                # Y_path の一つ目の Chunk をindex化して表示
                                if Y_path_chunk_number == 0:
                                    chunk_surface += N_index(Y_path_chunk, 'Y')
                                # X_path とぶつかったところまで戻ってきたら | として X_path に戻る
                                elif Y_path_chunk == X_path_chunk:
                                    chunk_surface += ' | '
                                    flag = 1
                                    break
                                else:
                                    chunk_surface += ' -> ' + chapter5_42.get_chunk_surface(Y_path_chunk)
                            chunk_surface += chapter5_42.get_chunk_surface(X_path_chunk)
                        # Y_path にぶつかる前　+　Y_path の表示が終わった後は X_path を表示
                        else:
                            chunk_surface += ' -> ' + chapter5_42.get_chunk_surface(X_path_chunk)
                    print(chunk_surface)

if __name__ == '__main__':
    main()