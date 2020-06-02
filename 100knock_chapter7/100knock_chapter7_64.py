import gensim

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin', binary=True)

    writefile = open('questions-words-result.txt', 'w', encoding='utf-8')
    with open('questions-words.txt', 'r', encoding='utf-8') as file:
        for line in file:
            words = line.split()
            if words[0] != ':':
                vec1 = words[0]
                vec2 = words[1]
                vec3 = words[2]
                most_word = model.most_similar(positive=[vec2, vec3], negative=[vec1], topn=1)
                writeline = line.strip() + ' ' + str(most_word[0][0]) + ' ' + str(most_word[0][1]) + '\n'
                writefile.write(writeline)
            else:
                writefile.write(line)
    writefile.close()

if __name__ == '__main__':
    main()