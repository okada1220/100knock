import re
import pprint
import collections
import matplotlib.pyplot as plt
import japanize_matplotlib

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

# 共起　=　隣接して出現する　として考える
words = []
for i in range(len(morph)):
    if morph[i]['base'] == '猫':
        for k in range(1,3):
            if i+k <= len(morph):
                words.append(morph[i+k]['base'])
            if i-k >= 0:
                words.append(morph[i-k]['base'])

frequency_word = collections.Counter(words)

most_frequency_word = frequency_word.most_common(10)
word = []
frequency = []
for i in range(10):
    word.append(most_frequency_word[i][0])
    frequency.append(most_frequency_word[i][1])

plt.bar(word, frequency, align='center')
plt.show()