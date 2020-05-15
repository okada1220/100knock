import re
import pprint
import collections
import matplotlib.pyplot as plt

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

words = []
for i in range(len(morph)):
    words.append(morph[i]['base'])

frequency_word = collections.Counter(words)

most_frequency_word = frequency_word.most_common()
level = []
frequency = []
for i in range(len(frequency_word)):
    level.append(i)
    frequency.append(most_frequency_word[i][1])

plt.xscale('log')
plt.yscale('log')
plt.plot(level, frequency)
plt.show()