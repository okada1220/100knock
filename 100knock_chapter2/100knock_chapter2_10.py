f = open('popular-names.txt', 'r', encoding='utf-8')
file = f.read()
f.close()

texts = file.split('\n')
texts.remove('')

print(len(texts))
