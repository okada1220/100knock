col1 = open('col1.txt', 'r', encoding='utf-8')
col2 = open('col2.txt', 'r', encoding='utf-8')

connect_text = col1.read() + '\t' + col2.read()
connect_col = open('connect_col.txt', 'w', encoding='utf-8')
connect_col.write(connect_text)
connect_col.close()

col1.close()
col2.close()