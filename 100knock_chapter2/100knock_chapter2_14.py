f = open('popular-names.txt', 'r', encoding='utf-8')
file = f.read()
f.close()

texts = file.split('\n')
texts.remove('')

print('何行目まで表示しますか？')
number = input()
N = int(number)

for text in texts[:N]:
    print(text)