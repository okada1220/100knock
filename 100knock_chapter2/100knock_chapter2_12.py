f = open('popular-names.txt', 'r', encoding='utf-8')
file = f.read()
f.close()

texts = file.split('\n')

col1 = open('col1.txt', 'w', encoding='utf-8')
col2 = open('col2.txt', 'w', encoding='utf-8')
col1.write(texts[0])
col2.write(texts[1])
col1.close()
col2.close()