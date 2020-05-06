f = open('popular-names.txt', 'r', encoding='utf-8')
file = f.read()
f.close()

texts = file.split('\n')
texts.remove('')

for i in range(len(texts)):
    texts[i] = texts[i].split('\t')

frequency_word = [[0 for i in range(2)] for j in range(len(texts))]
number = 0
for x in range(len(texts)):
    frag = 0
    for y in range(number):
        if texts[x][0] == frequency_word[y][0]:
            frequency_word[y][1] += 1
            frag = 1
    if frag == 0:
        frequency_word[number][0] = texts[x][0]
        frequency_word[number][1] = 1
        number += 1
print(frequency_word)

for n in range(len(frequency_word)-1, 0, -1):
    for m in range(0, n):
        if frequency_word[m][1] < frequency_word[m+1][1]:
            sub = frequency_word[m]
            frequency_word[m] = frequency_word[m+1]
            frequency_word[m+1] = sub
print(frequency_word)