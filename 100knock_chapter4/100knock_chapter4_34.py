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

longestNP = []
longestNP_len = 0
NP = ''
NP_len = 0
for i in range(len(map)):
    if map[i]['pos'] == '名詞':
        NP += map[i]['surface']
        NP_len += 1
    else:
        if NP_len > longestNP_len:
            longestNP = []
            longestNP.append(NP)
            longestNP_len = NP_len
            print(i)
        elif NP_len == longestNP_len:
            longestNP.append(NP)
            print(i)
        NP = ''
        NP_len = 0
print(longestNP)