import re
import pprint
import collections
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = 'C:\Users\naoki\OneDrive\デスクトップ\研究室関連\100本ノック\ipaexg.ttf'
font_prop = FontProperties(fname=font_path)

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

word = []
frequency = []
for i in range(10):
    print(frequency_word.most_common()[i])
    word.append(frequency_word.most_common()[i][0])
    frequency.append(frequency_word.most_common()[i][1])

plt.bar(word, frequency, align='center')
plt.show()