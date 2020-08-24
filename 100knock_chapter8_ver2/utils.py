import nltk  # nltk.download('all') 済み
import torch

# データファイルから特徴行列Ｘと正解ベクトルＹを返します
# 引数：ファイルへのパス、単語ベクトル変換用モデル、カテゴリ-ラベル対
def make_vector(filepath, vector_model, category_table):
    with open(filepath, 'r', encoding='utf-8') as file:
        x_matrix = []
        y_vector = []
        for line in file:
            category, title = line.split('\t')  # 各行からカテゴリ category 事例　title を得ます
            title = title.strip()
            words = nltk.word_tokenize(title)  # nltk を使って単語に分割します
            words_vector = torch.tensor([vector_model[word].tolist() for word in words if word in vector_model],
                                        dtype=torch.float)  # ベクトルに変換します  # モデルにない単語は無視します
            x_vector = torch.mean(words_vector, 0).unsqueeze(0)  # 平均Ｘを求めます  # unsqueeze は行列にまとめるための補正です
            x_matrix.append(x_vector)  # Ｘは x_matrix に入れときます
            y_vector.append(category_table[category])  # Yは y_vector に入れときます
        x_matrix = torch.cat(x_matrix, dim=0)  # x_matrix を行列に変換します
        y_vector = torch.tensor(y_vector)  # y_vector をベクトルに変換します
    return x_matrix, y_vector