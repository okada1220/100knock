import os
import torch
from torch import nn, optim
import numpy as np
import utils
import models

def main():
    # ファイル名を入力
    print('対象ファイルへのパスを入力して下さい。')
    filepath = input()
    while(filepath != ''):
        if os.path.exists(filepath):  # 例外処理
            x_matrix, y_vector = utils.load_vectorizer_file(filepath)
            x_matrix = torch.from_numpy(x_matrix)
            y_vector = torch.from_numpy(y_vector)

            INPUT_SIZE = 300
            OUTPUT_SIZE = 4
            singlenet_model = models.SingleNet_simple(INPUT_SIZE, OUTPUT_SIZE)
            loss = nn.CrossEntropyLoss()
            optimizer = optim.SGD(singlenet_model.parameters(), lr=0.1)

            pred_y_vector = singlenet_model.forward(x_matrix[0])
            l_loss = loss(pred_y_vector.unsqueeze(0), y_vector[0].unsqueeze(0))
            print(l_loss)

            EPOCH_NUM = 100
            for epoch in range(EPOCH_NUM):
                for i, x_vector in enumerate(x_matrix):
                    optimizer.zero_grad()
                    pred_y_vector = singlenet_model.forward(x_vector)
                    l_loss = loss(pred_y_vector.unsqueeze(0), y_vector[i].unsqueeze(0))
                    l_loss.backward()
                    optimizer.step()
            pred_y_vector = singlenet_model.forward(x_matrix[0])
            l_loss = loss(pred_y_vector.unsqueeze(0), y_vector[0].unsqueeze(0))
            print(l_loss)
            parameters = singlenet_model.parameters()
            for parameter in parameters:
                print(parameter)
                w_matrix = parameter.numpy()
                np.save('w_matrix.npy', w_matrix)
        else:
            print('対象ファイルが存在しません。\n')
        print('次の対象ファイルへのパスを入力して下さい。（終了：Enter）')  # 連続して処理を行います
        filepath = input()

if __name__ == '__main__':
    main()