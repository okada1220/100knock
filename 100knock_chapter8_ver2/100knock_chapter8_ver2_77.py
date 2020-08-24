import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import time
import models

def main():
    # ファイル名を入力
    print('対象ファイルへのパスを入力して下さい。')
    train_filepath = input()
    print('学習時のバッチサイズを入力して下さい。(整数のみ）')
    batchsize = int(input())

    if os.path.exists(train_filepath):  # 例外処理
        # 特徴行列Ｘと正解ベクトルＹを得ます
        train_x_matrix, train_y_vector = torch.load(train_filepath).values()

        # 訓練データをミニバッチ化します
        train_dataset = TensorDataset(train_x_matrix, train_y_vector)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

        # 単層ニューラルネットワークモデルを作ります
        INPUT_SIZE = 300
        OUTPUT_SIZE = 4
        singlenet_model = models.SingleNet_simple(INPUT_SIZE, OUTPUT_SIZE)
        loss = nn.CrossEntropyLoss()  # 損失関数を設定します
        optimizer = optim.SGD(singlenet_model.parameters(), lr=0.1)  # オプティマイザを設定します

        # 学習を行います
        EPOCH_NUM = 10
        for epoch in range(EPOCH_NUM):
            start_time = time.time()
            epoch_accuracy = 0
            for x_vector, y_vector in train_loader:
                optimizer.zero_grad()
                pred_y_vector = singlenet_model.forward(x_vector)
                l_loss = loss(pred_y_vector, y_vector)
                l_loss.backward()
                optimizer.step()
                epoch_accuracy += (y_vector == pred_y_vector.argmax(1)).sum().item()
            # 正解率・実行時間を計算します
            train_accuracy = epoch_accuracy / len(train_x_matrix)
            end_time = time.time()
            exe_time = end_time - start_time
            print('正解率：', train_accuracy, '\t実行時間：', exe_time)
    else:
        print('対象ファイルが存在しません。\n')
if __name__ == '__main__':
    main()

"""
<実行結果>

対象ファイルへのパスを入力して下さい。
train_vectorizer.pth
学習時のバッチサイズを入力して下さい。(整数のみ）
1
正解率： 0.863534256832647 	実行時間： 4.464057683944702
正解率： 0.8979782852864095 	実行時間： 4.395245313644409
正解率： 0.904717334331711 	実行時間： 4.616651773452759
正解率： 0.9093972295020591 	実行時間： 4.289527893066406
正解率： 0.9120179707974542 	実行時間： 4.60470986366272
正解率： 0.9147323099962561 	実行時間： 4.62859582901001
正解率： 0.9152938974166979 	実行時間： 4.325431823730469
正解率： 0.9163234743541745 	実行時間： 4.547870635986328
正解率： 0.9165106701609884 	実行時間： 4.380272150039673
正解率： 0.917446649195058 	実行時間： 4.46705436706543

Process finished with exit code 0

対象ファイルへのパスを入力して下さい。
train_vectorizer.pth
学習時のバッチサイズを入力して下さい。(整数のみ）
2
正解率： 0.8442530887308124 	実行時間： 2.3886139392852783
正解率： 0.8898352676900038 	実行時間： 2.236018657684326
正解率： 0.8983526769000374 	実行時間： 2.230034828186035
正解率： 0.9034069636840135 	実行時間： 2.2998499870300293
正解率： 0.9079932609509547 	実行時間： 2.259956121444702
正解率： 0.9101460127293148 	実行時間： 2.211085796356201
正解率： 0.9102396106327219 	実行時間： 2.2818965911865234
正解率： 0.9126731561213028 	実行時間： 2.265939474105835
正解率： 0.9130475477349307 	実行時間： 2.2908740043640137
正解率： 0.9144515162860352 	実行時間： 2.408557176589966

Process finished with exit code 0

対象ファイルへのパスを入力して下さい。
train_vectorizer.pth
学習時のバッチサイズを入力して下さい。(整数のみ）
4
正解率： 0.8192624485211532 	実行時間： 1.326453685760498
正解率： 0.8757955821789591 	実行時間： 1.3444042205810547
正解率： 0.8887120928491202 	実行時間： 1.2626311779022217
正解率： 0.8936727817296892 	実行時間： 1.220726490020752
正解率： 0.8984462748034444 	実行時間： 1.2696056365966797
正解率： 0.9009734181954324 	実行時間： 1.3115012645721436
正解率： 0.9029389741669787 	実行時間： 1.2147409915924072
正解率： 0.9056533133657806 	実行時間： 1.3234617710113525
正解率： 0.9081804567577686 	実行時間： 1.3703529834747314
正解率： 0.9084612504679895 	実行時間： 1.3493716716766357

Process finished with exit code 0

対象ファイルへのパスを入力して下さい。
train_vectorizer.pth
学習時のバッチサイズを入力して下さい。(整数のみ）
8
正解率： 0.7953013852489704 	実行時間： 0.7689120769500732
正解率： 0.8507113440658929 	実行時間： 0.7120952606201172
正解率： 0.8717708723324598 	実行時間： 0.6712040901184082
正解率： 0.8820666417072257 	実行時間： 0.7260911464691162
正解率： 0.8897416697865967 	実行時間： 0.6881275177001953
正解率： 0.8924560089853987 	実行時間： 0.6392884254455566
正解率： 0.8939535754399102 	実行時間： 0.7180807590484619
正解率： 0.8957319356046425 	実行時間： 0.7300465106964111
正解率： 0.8976038936727817 	実行時間： 0.6951422691345215
正解率： 0.8980718831898166 	実行時間： 0.6652233600616455

Process finished with exit code 0
"""