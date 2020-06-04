import gensim
import re
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import japanize_matplotlib

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', binary=True)

    with open('country.txt', 'r', encoding='utf-8') as file:
        file_country_names = [re.sub(' ', '_', line.strip()) for line in file]

    # モデルに含まれていない国名は無視します
    country_names = [name for name in file_country_names if name in model]

    country_vectors = [model[i] for i in country_names]

    country_vectors = np.array(country_vectors)
    pred = linkage(country_vectors, method='ward')

    dendrogram(pred, labels=country_names)
    plt.show()

if __name__ == '__main__':
    main()