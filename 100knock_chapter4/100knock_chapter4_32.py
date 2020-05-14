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

for i in range(len(map)):
    if map[i]['pos'] == '動詞':
        print(map[i]['base'])