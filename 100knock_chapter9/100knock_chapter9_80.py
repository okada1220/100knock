import collections

# file から word と id の対応 dict を作ります
def get_word_id_dict(filename):
    # 単語と出現回数をもつ　words_frequency を作ります
    words_frequency = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            words = title.strip().split()
            for word in words:
                words_frequency[word] += 1
    words_frequency = sorted(words_frequency.items(), key=lambda x: x[1], reverse=True)

    # 対応　dict として word_id_dict を作ります
    word_id_dict = {}
    id = 1
    # 出現回数が高いトップ５はストップワードとして id に 0 を割り当てます
    for i, word_frequency in enumerate(words_frequency):
        word = word_frequency[0]
        frequency = word_frequency[1]
        if frequency > 1 and i > 4:
            word_id_dict[word] = id
            id += 1
        else:
            word_id_dict[word] = 0
    return word_id_dict

def main():
    word_id_dict = get_word_id_dict('../100knock_chapter6/train.feature.txt')

    with open('../100knock_chapter6/train.feature.txt', 'r', encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            words = title.strip().split()
            # 一応、未知語には 0 を割り当てるようにしてます
            words_id = [word_id_dict.get(word, 0) for word in words]
            print(words)
            print(words_id)

    with open('../100knock_chapter6/test.feature.txt', 'r',encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            words = title.strip().split()
            # 一応、未知語には 0 を割り当てるようにしてます
            words_id = [word_id_dict.get(word, 0) for word in words]
            print(words)
            print(words_id)

if __name__ == '__main__':
    main()