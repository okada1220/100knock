def n_gram(target, n):
    n_gram_words = []
    for i in range(len(target)-n+1):
        n_gram_words += [target[i:i+n]]
    return n_gram_words

text_X = 'paraparaparadise'
text_Y = 'paragraph'

X = n_gram(text_X, 2)
Y = n_gram(text_Y, 2)
print(X)
print(Y)

sum = []
pro = []
for word in X:
    if word in Y:
        pro.append(word)
    else:
        sum.append(word)
sum += Y
print(sum)
print(pro)

X_dif = [w for w in X if w not in Y]
Y_dif = [w for w in Y if w not in X]
print(X_dif)
print(Y_dif)

if 'se' in X:
    print('Xの中にseが含まれています。')
else:
    print('Xの中にseが含まれていません。')

if 'se' in Y:
    print('Yの中にseが含まれています。')
else:
    print('Yの中にseが含まれていません。')