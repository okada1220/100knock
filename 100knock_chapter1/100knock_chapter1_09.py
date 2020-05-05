import random

print('textを入力してください。')
text = input()

words = text.split()

changed_text = ''
for word in words:
    if len(word) > 4:
        middle_word = [w for w in word[1:len(word)-1]]
        random.shuffle(middle_word)
        middle_words = ''
        for i in range(len(middle_word)):
            middle_words += middle_word[i]
        changed_text += word[0] + middle_words + word[len(word)-1] + ' '
    else:
        changed_text += word + ' '

print(changed_text)