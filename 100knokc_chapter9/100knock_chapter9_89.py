from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

# BERT 用にデータを inputs_list と category_list に分けます
# inputs_list には各単語が数値として格納された input_ids とパディングされた部分かどうかを判別する attention_mask があります
# category_list には各カテゴリが入ってます
def make_dataset(filename, tokenizer, category_table, maxlen):
    inputs_list = []
    category_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            category, title = line.split('\t')
            words = title.strip().split()
            # BERT の tokenizer で単語列を変換します
            inputs = tokenizer.encode_plus(words, add_special_tokens=True, max_length=maxlen, pad_to_max_length=True)
            inputs_list.append(inputs)
            category_list.append(category_table[category])
    return inputs_list, category_list

# BERT のモデルを作ります
class BERTClass(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(768, output_size)

    def forward(self, inputs):
        # inputs の中から BERT の引数に合わせた形に変換します
        ids = torch.unsqueeze(torch.LongTensor(inputs['input_ids']), 0)
        mask = torch.unsqueeze(torch.LongTensor(inputs['attention_mask']), 0)
        _, output = self.bert(ids, attention_mask=mask)
        output = self.out(output)
        return output

def main():
    # BERT の Tokenizer を読み込みます
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # category を対応する数字にします
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}

    # 学習データを読み込みます
    inputs_list, category_list = make_dataset('../100knock_chapter6/minimini-train.feature.txt', tokenizer, category_table, 20)

    # BERT のモデルを作ります
    OUTPUT_SIZE = 4
    model = BERTClass(OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # エポック回数３回で学習させます
    EPOCH_NUM = 3
    for epoch in range(EPOCH_NUM):
        loss_sum = 0
        accuracy_sum = 0
        for i in range(len(inputs_list)):
            # 初期化します
            optimizer.zero_grad()
            # 各事例で損失を求め、重みを更新します
            inputs = inputs_list[i]
            output = torch.LongTensor([category_list[i]])
            pred_output = model.forward(inputs)
            loss = criterion(pred_output, output)
            loss.backward()
            optimizer.step()
            loss_sum += loss
            accuracy_sum += int(category_list[i] == torch.argmax(pred_output))
        # 正答率と損失の平均を求めます
        accuracy = accuracy_sum / len(inputs_list)
        print('accuracy(train):', accuracy)
        loss_average = loss_sum / len(inputs_list)
        print('loss(train):', loss_average)

    # 評価データを読み込みます
    inputs_list, category_list = make_dataset('../100knock_chapter6/mini-test.feature.txt', tokenizer, category_table, 20)

    # 初期化します
    accuracy_sum = 0
    loss_sum = 0
    for i in range(len(inputs_list)):
        inputs = inputs_list[i]
        output = torch.LongTensor([category_list[i]])
        pred_output = model.forward(inputs)
        loss = criterion(pred_output, output)
        loss_sum += loss
        accuracy_sum += int(category_list[i] == torch.argmax(pred_output))
    # 正答率と損失の平均を求めます
    accuracy = accuracy_sum / len(inputs_list)
    print('accuracy(test):', accuracy)
    loss_average = loss_sum / len(inputs_list)
    print('loss(test):', loss_average)

    # minimini-train.feature.txtでの学習結果（100文）
    # accuracy(train): 0.31
    # loss(train): tensor(1.7046, grad_fn=<DivBackward0>)
    # accuracy(train): 0.29
    # loss(train): tensor(1.5366, grad_fn=<DivBackward0>)
    # accuracy(train): 0.3
    # loss(train): tensor(1.4808, grad_fn=<DivBackward0>)

    # mini-test.feature.txtでのテスト結果（500文）
    # accuracy(test): 0.424
    # loss(test): tensor(1.1717, grad_fn=<DivBackward0>)

if __name__ == '__main__':
    main()