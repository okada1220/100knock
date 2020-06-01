import gensim
import pprint

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Users\naoki\Downloads\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', binary=True)
    model.save(r'C:\Users\naoki\Documents\models\model.bin')
    pprint.pprint(model['United_States'])

if __name__ == '__main__':
    main()