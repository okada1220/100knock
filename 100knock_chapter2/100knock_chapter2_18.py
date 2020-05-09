f = open('popular-names.txt', 'r', encoding='utf-8')
file = f.read()
f.close()

texts = file.split('\n')
texts.remove('')

for i in range(len(texts)):
    texts[i] = texts[i].split('\t')
print(texts)

texts.sort(key=lambda x: int(x[2]), reverse=True)
print('ソート後')
print(texts)