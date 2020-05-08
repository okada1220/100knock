col1 = open('col1.txt', 'w', encoding='utf-8')
col2 = open('col2.txt', 'w', encoding='utf-8')

with open('popular-names.txt', 'r', encoding='utf-8') as file:
    for texts in file:
        text = texts.split()
        col1.write(text[0] + '\n')
        col2.write(text[1] + '\n')

col1.close()
col2.close()