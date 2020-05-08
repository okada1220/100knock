def n_gram(target, n):
    n_gram_words = []
    for i in range(len(target)-n+1):
        n_gram_words += [target[i:i+n]]
    return n_gram_words

text_X = 'paraparaparadise'
text_Y = 'paragraph'

X = n_gram(text_X, 2)
Y = n_gram(text_Y, 2)
X_group = set(X)
Y_group = set(Y)

sum = X_group.union(Y_group)
pro = X_group.intersection(Y_group)
print(sum)
print(pro)

X_dif = X_group.difference(Y_group)
Y_dif = Y_group.difference(X_group)
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