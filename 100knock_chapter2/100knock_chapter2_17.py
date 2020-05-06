f = open('popular-names.txt', 'r', encoding='utf-8')
file = f.read()
f.close()

texts = file.split('\n')
texts.remove('')

for i in range(len(texts)):
    texts[i] = texts[i].split('\t')

unique_words = []
for x in range(len(texts)):
    if texts[x][0] not in unique_words:
        unique_words.append(texts[x][0])
print(unique_words)