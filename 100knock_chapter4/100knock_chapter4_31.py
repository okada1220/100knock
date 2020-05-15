import re
import pprint
mgs = __import__('100knock_chapter4_30')

def main():
    morph = mgs.make_morph('neko.txt.mecab')

    for i in range(len(morph)):
        if morph[i]['pos'] == '動詞':
            print(morph[i]['surface'])

if __name__ == '__main__':
    main()