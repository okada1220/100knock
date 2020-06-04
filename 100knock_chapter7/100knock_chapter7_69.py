import gensim
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', binary=True)

    with open('country.txt', 'r', encoding='utf-8') as file:
        file_country_names = [re.sub(' ', '_', line.strip()) for line in file]

    # モデルに含まれていない国名は無視します
    country_names = [name for name in file_country_names if name in model]

    country_vectors = [model[i] for i in country_names]

    kmeans_country_vectors = np.array(country_vectors)
    kmeans_pred = KMeans(n_clusters=5).fit_predict(kmeans_country_vectors)
    colortable =  {0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'black'}
    color = [colortable[pred] for pred in kmeans_pred]

    tsne_pred = TSNE().fit_transform(country_vectors)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(tsne_pred[:, 0], tsne_pred[:, 1], c=color[:])
    plt.show()

if __name__ == '__main__':
    main()