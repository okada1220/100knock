import gensim
import re
import numpy as np
from sklearn.cluster import KMeans

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', binary=True)

    with open('country.txt', 'r', encoding='utf-8') as file:
        file_country_names = [re.sub(' ', '_', line.strip()) for line in file]

    # モデルに含まれていない国名は無視します
    country_names = [name for name in file_country_names if name in model]

    country_vectors = [model[i] for i in country_names]

    country_vectors = np.array(country_vectors)
    preds = KMeans(n_clusters=5).fit_predict(country_vectors)

    for country_name, pred in zip(country_names, preds.tolist()):
        print(country_name, pred)

if __name__ == '__main__':
    main()