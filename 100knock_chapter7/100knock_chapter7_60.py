import gensim
import pprint

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', binary=True)
    pprint.pprint(model['United_States'])

if __name__ == '__main__':
    main()