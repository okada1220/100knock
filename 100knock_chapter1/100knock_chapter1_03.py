import re

text = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
text = re.sub(',|\.', '', text)
words = text.split()
word_lengths = [len(word) for word in words]
print(word_lengths)