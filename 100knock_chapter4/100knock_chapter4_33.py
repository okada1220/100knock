import re
import pprint

morph = []
with open('neko.txt.mecab', 'r', encoding='utf-8') as f:
    for line in f:
        if not line == 'EOS\n':
            info = re.split(r'\t|,', line)
            key_list = {}
            key_list['surface'] = info[0]
            key_list['base'] = info[7]
            key_list['pos'] = info[1]
            key_list['pos1'] = info[2]
            morph.append(key_list)

for i in range(len(morph)-2):
    if morph[i+1]['surface'] == 'の' and morph[i]['pos'] == '名詞' and morhp[i+2]['pos'] == '名詞':
        NP = ''
        for k in range(3):
            NP += morph[i+k]['surface']
        print(NP)