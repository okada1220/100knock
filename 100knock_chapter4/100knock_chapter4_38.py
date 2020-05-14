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

words = []
for i in range(len(map)):
    words.append(map[i]['base'])

frequency_word = collections.Counter(words)

frequency = []
for i in range(len(frequency_word)):
    frequency.append(frequency_word.most_common()[i][1])

plt.hist(frequency)
plt.show()

plt.hist(frequency, range=(0, 20), bins=20)
plt.show()