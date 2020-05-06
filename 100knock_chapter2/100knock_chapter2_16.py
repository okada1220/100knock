f = open('popular-names.txt', 'r', encoding='utf-8')
file = f.read()
f.close()

texts = file.split('\n')
texts.remove('')

print('何分割しますか？')
number = input()
N = int(number)
gap = len(texts) / N

for i in range(N):
    filename = number + 'split-popular-names{}.txt'.format(i+1)
    w = open(filename, 'w', encoding='utf-8')
    first = int(i*gap)
    last = int((i+1)*gap)
    for text in texts[first:last]:
        w.write(text + '\n')
    w.close()