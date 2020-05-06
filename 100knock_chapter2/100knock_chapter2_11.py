import re

f = open('popular-names.txt', 'r', encoding='utf-8')
file = f.read()
f.close()

changed_file = re.sub('\t', ' ', file)

texts = file.split('\n')
changed_texts = changed_file.split('\n')

for i in range(10):
    print(texts[i])
    print(changed_texts[i])