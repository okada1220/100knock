import re
import pprint

def make_morph(filename):
    morph = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line == 'EOS\n':
                info = re.split(r'\t|,', line)
                key_list = {}
                key_list['surface'] = info[0]
                key_list['base'] = info[7]
                key_list['pos'] = info[1]
                key_list['pos1'] = info[2]
                morph.append(key_list)
    return morph

def main():
    morph = make_morph('neko.txt.mecab')
    pprint.pprint(morph)

if __name__ == '__main__':
    main()