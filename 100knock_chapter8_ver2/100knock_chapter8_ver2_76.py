import os
import torch
from torch import nn, optim
import models

def main():
    # ファイル名を入力
    print('訓練データファイルへのパスを入力して下さい。')
    train_filepath = input()

    if os.path.exists(train_filepath):  # 例外処理
        # 特徴行列Ｘと正解ベクトルＹを得ます
        train_x_matrix, train_y_vector = torch.load(train_filepath).values()

        # 単層ニューラルネットワークモデルを作ります
        INPUT_SIZE = 300
        OUTPUT_SIZE = 4
        singlenet_model = models.SingleNet_simple(INPUT_SIZE, OUTPUT_SIZE)
        loss = nn.CrossEntropyLoss()  # 損失関数を設定します
        optimizer = optim.SGD(singlenet_model.parameters(), lr=0.1)  # オプティマイザを設定します

        # 学習を行います
        EPOCH_NUM = 100
        for epoch in range(EPOCH_NUM):
            for i, x_vector in enumerate(train_x_matrix):
                optimizer.zero_grad()
                pred_y_vector = singlenet_model.forward(x_vector)
                l_loss = loss(pred_y_vector.unsqueeze(0), train_y_vector[i].unsqueeze(0))
                l_loss.backward()
                optimizer.step()
            # 各エポックごとのチェックポイントを保存します
            torch.save({'epoch':epoch,
                        'model_state_dict':singlenet_model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict()},
                       './checkpoint_data/epoch' + str(epoch) + '.pth')
    else:
        print('対象ファイルが存在しません。\n')

if __name__ == '__main__':
    main()