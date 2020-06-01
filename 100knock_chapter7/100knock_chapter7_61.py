import gensim

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('models/model.bin', binary=True)
    model.similarity('United_States', 'U.S.')

if __name__ == '__main__':
    main()