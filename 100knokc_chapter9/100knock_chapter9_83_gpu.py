import collections
import torch
import torch.nn as nn

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

class MyRNN2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        # Linear を使って単語の one-hot ベクトルから埋め込みベクトルを得ます
        # -> Embedding を使った方がいい？（入力の形を変更する必要あり）（保留中）
        self.input = nn.Linear(input_size, hidden_size1)
        # RNN の活性化関数には tanh を使います
        self.rnn = nn.RNN(hidden_size1, hidden_size2, num_layers=1, nonlinearity='tanh')
        self.out = nn.Linear(hidden_size2, output_size)

    def forward(self, input):
        input = self.input(input)
        output, self.hidden = self.rnn(input, self.hidden)
        output = self.out(output)
        return output

    # 隠れ状態を初期化します
    def reset(self):
        self.hidden = torch.zeros(1, 1, self.hidden_size2)

def main():
    # 単語のＩＤ化に 80 のプログラムを使い、対応表として辞書 word_id_dict を得ます
    word_id_dict = get_word_id_dict('/content/drive/My Drive/100knock_chapter6/train.feature.txt')
    vocab = max(word_id_dict.values()) + 1

    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    # RNN を作って、隠れ状態ベクトルを初期化します
    INPUT_SIZE = vocab
    HIDDEN_SIZE1 = 300
    HIDDEN_SIZE2 = 50
    OUTPUT_SIZE = 4
    print('OK1')
    model = MyRNN2(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE).cuda()
    print('OK2')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    EPOCH_NUM = 3
    BATCH_SIZE = 10
    flag = 0
    for epoch in range(EPOCH_NUM):
        accuracy_sum = 0
        loss_sum = 0
        loss_batch_sum = 0
        counter = 0
        print('OK3')
        with open('/content/drive/My Drive/100knock_chapter6/minimini-train.feature.txt', 'r', encoding='utf-8') as file:
          for line in file:
            print('OK4')
            category, title = line.split('\t')
            words = title.strip().split()
            model.reset()
            optimizer.zero_grad()
            print('OK5')
            words_vector = torch.stack([torch.unsqueeze(torch.eye(max(word_id_dict.values())+1)[word_id_dict.get(word, 0)], 0) for word in words], dim=0)
            print('OK5.5')
            category_vector = torch.Tensor([category_table[category]])
            print('OK6')
            words_vector = words_vector.cuda()
            category_vector = category_vector.cuda()
            print('OK7')
            pred_category_vector = model.forward(words_vector)
            loss = criterion(pred_category_vector[-1], category_vector.type(torch.long))
            loss_batch_sum += loss
            flag += 1
            if flag >= BATCH_SIZE:
                loss = loss_batch_sum / BATCH_SIZE
                loss.backward()
                print(loss)
                optimizer.step()
                loss_batch_sum = 0
                flag = 0
            loss_sum += loss
            accuracy_sum += int(category_vector == torch.argmax(pred_category_vector[-1][-1]))
            counter += 1
        accuracy = accuracy_sum / counter
        print('accuracy(train):', accuracy)
        loss_average = loss_sum / counter
        print('loss(train):', loss_average)

    accuracy_sum = 0
    loss_sum = 0
    counter = 0
    model.reset()
    with open('/content/drive/My Drive/100knock_chapter6/mini-test.feature.txt', 'r', encoding='utf-8') as file:
      for line in file:
        category, title = line.split('\t')
        words = title.strip().split()
        words_vector = torch.stack([torch.unsqueeze(torch.eye(max(word_id_dict.values())+1)[word_id_dict.get(word, 0)], 0) for word in words], dim=0)
        category_vector = torch.Tensor([category_table[category]])
        words_vector = words_vector.cuda()
        category_vector = category_vector.cuda()
        pred_category_vector = model.forward(words_vector)
        loss = criterion(pred_category_vector[-1], category_vector.type(torch.long))
        loss_sum += loss
        accuracy_sum += int(category_vector == torch.argmax(pred_category_vector[-1][-1]))
        counter += 1
    accuracy = accuracy_sum / counter
    print('accuracy(test):', accuracy)
    loss_average = loss_sum / counter
    print('loss(test):', loss_average)

    # 実行結果
    # OK1
    # OK2
    # OK3
    # OK4
    # OK5
    # -> ここでセッションがクラッシュするみたいです

if __name__ == '__main__':
    main()