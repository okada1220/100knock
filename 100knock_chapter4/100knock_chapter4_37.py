import re
import pprint
import collections
import matplotlib.pyplot as plt

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

# 共起　=　隣接して出現する　として考える
words = []
for i in range(len(map)):
    if map[i]['base'] == '猫':
        for k in range(1,3):
            if i+k <= len(map):
                words.append(map[i+k]['base'])
            if i-k >= 0:
                words.append(map[i-k]['base'])

frequency_word = collections.Counter(words)

word = []
frequency = []
for i in range(10):
    print(frequency_word.most_common()[i])
    word.append(frequency_word.most_common()[i][0])
    frequency.append(frequency_word.most_common()[i][1])

plt.bar(word, frequency, align='center')
plt.show()