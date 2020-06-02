import gensim
import pandas as pd
from scipy.stats import spearmanr

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', binary=True)

    df = pd.read_csv('wordsim353/combined.csv', encoding='utf-8')
    wordset_list = [{'word1':df['Word 1'][index],
                  'word2':df['Word 2'][index],
                  'human_score':df['Human (mean)'][index],
                  'model_score':model.similarity(df['Word 1'][index], df['Word 2'][index]),
                  'human_rank':0,
                  'model_rank':0} for index in df.index]
    wordset_list.sort(key=lambda x: x['human_score'], reverse=True)
    for i, wordset in enumerate(wordset_list):
        wordset['human_rank'] = i+1
    wordset_list.sort(key=lambda x: x['model_score'], reverse=True)
    for i, wordset in enumerate(wordset_list):
        wordset['model_rank'] = i+1

    human_ranklist = [wordset['human_rank'] for wordset in wordset_list]
    model_ranklist = [wordset['model_rank'] for wordset in wordset_list]
    correlation, pvalue = spearmanr(human_ranklist, model_ranklist)
    print('相関係数', correlation)
    print('p値', pvalue)


if __name__ == '__main__':
    main()