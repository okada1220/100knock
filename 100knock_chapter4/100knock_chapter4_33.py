import re
import pprint

map = []
with open('neko.txt.mecab', 'r', encoding='utf-8') as f:
    for line in f:
        if not line == 'EOS\n':
            info = re.split(r'\t|,', line)
            list = {}
            list['surface'] = info[0]
            list['base'] = info[7]
            list['pos'] = info[1]
            list['pos1'] = info[2]
            map.append(list)

for i in range(len(map)-2):
    if map[i+1]['surface'] == 'の' and map[i]['pos'] == '名詞' and map[i+2]['pos'] == '名詞':
        NP = ''
        for k in range(3):
            NP += map[i+k]['surface']
        print(NP)