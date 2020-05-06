f = open('popular-names.txt', 'r', encoding='utf-8')
file = f.read()
f.close()

texts = file.split('\n')
texts.remove('')

for i in range(len(texts)):
    texts[i] = texts[i].split('\t')
print(texts)

for x in range(len(texts)-1, 0, -1):
    for y in range(0, x):
        if int(texts[y][2]) < int(texts[y+1][2]):
            sub = texts[y]
            texts[y] = texts[y+1]
            texts[y+1] = sub
print('ソート後')
print(texts)