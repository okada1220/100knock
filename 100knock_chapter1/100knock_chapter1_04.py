import re

text = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
text = re.sub(',|\.', '', text)
words = text.split()

number = 0
word_dic = {}

for word in words:
    number += 1
    if number==1 or 5<=number<10 or 15<=number<17 or number==19:
        word_dic[word[0]] = number
    else:
        word_dic[word[0:2]] = number

print(word_dic)