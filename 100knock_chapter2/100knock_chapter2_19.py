import collections

text_name = []
with open('popular-names.txt', 'r', encoding='utf-8') as file:
    for texts in file:
        text = texts.split()
        text_name.append(text[0])

frequency_word = collections.Counter(text_name)
print(frequency_word)

frequency_word.most_common()
print(frequency_word)