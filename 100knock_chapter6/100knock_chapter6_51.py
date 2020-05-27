import MeCab
import re

# filename からカテゴリと記事見出しの単語分割を feature_filename に保存します
def make_featurefile(filename, feature_filename):
    mecab = MeCab.Tagger("-Owakati")
    write_file = open(feature_filename, 'w', encoding='utf-8')
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            category, title = line.split('\t')
            title = re.sub('\n', '', title)
            words = mecab.parse(title)
            write_file.write(category + '\t' + words)
    write_file.close()

def main():
    make_featurefile('train.txt', 'train.feature.txt')
    make_featurefile('valid.txt', 'valid.feature.txt')
    make_featurefile('test.txt', 'test.feature.txt')

if __name__ == '__main__':
    main()