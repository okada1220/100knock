import pandas as pd
from sklearn.model_selection import train_test_split

# 引数となる filename から各カテゴリの事例数を表示します
def count_category(filename):
    category_count = {'b': 0, 't': 0, 'e': 0, 'm': 0}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            category = line.split('\t')[0]
            category_count[category] += 1
    print(filename + str(category_count))

def main():
    # csv ファイルを読み込んでヘッダーを加えます
    df = pd.read_csv('NewsAggregatorDataset/newsCorpora.csv',
                     names=('ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'),
                     header=None,  sep='\t|\n', encoding='utf-8')

    articles_data = []
    publisher = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']  # 抽出する情報源
    for i, df_publisher in enumerate(df['PUBLISHER']):
        if df_publisher in publisher:
            articles_data.append(df['CATEGORY'][i] + '\t' + df['TITLE'][i])

    # train_data=0.8, ,valid_data=0.1, test_data=0.1 に分割します
    train_data, valid_test_data = train_test_split(articles_data,
                                                   train_size=0.8,
                                                   random_state=42)
    valid_data, test_data = train_test_split(valid_test_data,
                                             test_size=0.5,
                                             random_state=42)

    # ファイルを作成します
    with open('train.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_data))
    with open('valid.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_data))
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_data))

    # 各データの各カテゴリの事例数を表示します
    count_category('train.txt')
    count_category('test.txt')

if __name__ == '__main__':
    main()