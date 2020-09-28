# -*- coding: utf-8 -*-

import nltk
import collections
import torch

def make_word_id_dict(filepath):
    words_frequency = collections.defaultdict(int)
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            title = title.strip()
            words = nltk.word_tokenize(title)
            for word in words:
                words_frequency[word] += 1
    words_frequency = sorted(words_frequency.items(),
                             key=lambda x: x[1],
                             reverse=True)

    word_id_dict = {}
    for i, words_frequency in enumerate(words_frequency):
        word, frequency = words_frequency
        if frequency > 1:
            word_id_dict[word] = i+1
        else:
            word_id_dict[word] = 0

    return word_id_dict

def trans_words_id(word_id_dict, sentence):
    sentence = sentence.strip()
    words = nltk.word_tokenize(sentence)
    words_id = torch.tensor([word_id_dict.get(word, 0) for word in words])

    return words_id

def split_category_title(filepath, category_table, word_id_dict):
    categories = []
    titles = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            categories.append(category)
            titles.append(title)
    categories = torch.tensor([category_table[category] for category in categories])
    titles = [trans_words_id(word_id_dict, title) for title in titles]

    return categories, titles