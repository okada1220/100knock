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

word = []
frequency = []
for i in range(10):
    print(frequency_word.most_common()[i])
    word.append(frequency_word.most_common()[i][0])
    frequency.append(frequency_word.most_common()[i][1])

plt.bar(word, frequency, align='center')
plt.show()