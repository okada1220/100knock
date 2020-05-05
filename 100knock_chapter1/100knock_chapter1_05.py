def n_gram(target, n):
    n_gram_words = []
    for i in range(len(target)-n+1):
        n_gram_words += [target[i:i+n]]
    return n_gram_words

text = 'I am an NLPer'

text_bi_gram = n_gram(text, 2)
print(text_bi_gram)

word = text.split()
word_bi_gram = n_gram(word, 2)
print(word_bi_gram)