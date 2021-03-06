# -*- coding: utf-8 -*-

import sys
import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import utils
import models

"""
TensorDataset, DataLoader を使う場合
"""

def main():
    # コマンドライン引数から必要な情報を得ます
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    if len(sys.argv) < 4:
        print('python 100knock_chapter9_ver2_83(1) [dict_filepath] [train_filepath] [batch_size]')
        sys.exit()
    filepath = sys.argv[1]
    train_filepath = sys.argv[2]
    batch_size = sys.argv[3]
    while (1):
        if os.path.exists(filepath) and os.path.exists(train_filepath) and batch_size.isdigit():
            word_id_dict = utils.make_word_id_dict(filepath)
            train_category, train_title = utils.split_category_title(train_filepath, category_table, word_id_dict)
            break
        if not os.path.exists(filepath):
            print('対象の辞書となるファイルが存在しません。')
            print('もう一度、入力して下さい。')
            filepath = input()
        if not os.path.exists(train_filepath):
            print('訓練データの対象ファイルが存在しません。')
            print('もう一度、入力して下さい。')
            train_filepath = input()
        if not batch_size.isdigit():
            print('バッチサイズが正しくありません。')
            print('学習時のバッチサイズを入力して下さい。（整数のみ）')
            batch_size = input()
    batch_size = int(batch_size)

    # 全てのデータ（タイトル）を最大長でパディングします
    max_length = 0
    for title in train_title:
        if max_length < len(title):
            max_length = len(title)
    padding_id = max(word_id_dict.values()) + 1
    padding_train_title = torch.tensor([[len(title)]
                                        + title.tolist()
                                        + [padding_id] * (max_length - len(title))
                                        for title in train_title])
    train_dataset = TensorDataset(padding_train_title, train_category)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # RNNモデルを作ります
    input_size = padding_id + 1
    embed_size = 300
    hidden_size = 50
    output_size = 4
    model = models.RNN_simple_minibatch_gpu(input_size, embed_size, hidden_size, output_size, padding_id).cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 訓練を行います
    epoch_num = 10
    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_accuracy = 0
        for title, category in train_loader:
            optimizer.zero_grad()
            model.reset(batch_size=len(title))
            title = title.cuda()
            category = category.cuda()
            title_len = title[:, 0]
            title_con = title[:, 1:]
            pred_train_category = model.forward(title_con, title_len)
            train_loss = loss(pred_train_category, category)
            train_loss.backward()
            optimizer.step()
            epoch_loss.append(train_loss.item())
            epoch_accuracy += (category == pred_train_category.argmax(1)).sum().item()
        average_loss = sum(epoch_loss) / len(epoch_loss)
        accuracy_rate = epoch_accuracy / len(train_category)
        print('損失：', average_loss, '\t正解率：', accuracy_rate)
if __name__ == '__main__':
    main()

"""
<実行結果（仮）>

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
1
損失： 2.694353051063309 	正解率： 0.43157993260950955
損失： 2.7524300550733503 	正解率： 0.43962935230250844
損失： 2.7859612264319003 	正解率： 0.4338262822912767
損失： 2.7740175121906363 	正解率： 0.43850617746162485
損失： 2.730507904039219 	正解率： 0.44805316360913516
損失： 2.7250115347663146 	正解率： 0.44721078247847246
損失： 2.691476455176062 	正解率： 0.4553538000748783
損失： 2.6298469254284385 	正解率： 0.47163983526769
損失： 2.678865216777374 	正解率： 0.45891052040434294
損失： 2.6332205269813653 	正解率： 0.47098464994384126

Process finished with exit code 0

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
2
損失： 1.4796752191316238 	正解率： 0.5068326469487083
損失： 1.4175403195642133 	正解率： 0.5584050917259453
損失： 1.3785013391569452 	正解率： 0.5707600149756645
損失： 1.346030430927914 	正解率： 0.5819917633845002
損失： 1.678669786653424 	正解率： 0.45862972669412205
損失： 1.6682753618058248 	正解率： 0.46433919880194685
損失： 1.6707661883039584 	正解率： 0.4692998876825159
損失： 1.6556881251982245 	正解率： 0.4722950205915388
損失： 1.6173897621877182 	正解率： 0.4883002620741295
損失： 1.5971426184192226 	正解率： 0.4938225383751404

Process finished with exit code 0

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
4
損失： 1.1122410321745566 	正解率： 0.5893859977536503
損失： 0.9824457830302794 	正解率： 0.662392362411082
損失： 0.9131102778215571 	正解率： 0.6868214152002995
損失： 0.89857287025508 	正解率： 0.6951516286035193
損失： 0.9130173329966863 	正解率： 0.6807375514788468
損失： 0.9404962909722073 	正解率： 0.6722201422688132
損失： 0.8662946789990239 	正解率： 0.7096593036315987
損失： 0.9072748056987042 	正解率： 0.6816735305129165
損失： 0.8549919639459875 	正解率： 0.7057281916885062
損失： 0.8125365370565845 	正解率： 0.7208910520404342

Process finished with exit code 0

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
8
損失： 1.0070121428946 	正解率： 0.6241108199176338
損失： 0.8249429189181792 	正解率： 0.7112504679895171
損失： 0.7267265383204344 	正解率： 0.7459752901535006
損失： 0.6786545334372692 	正解率： 0.7557094721078248
損失： 0.6692641537168069 	正解率： 0.7645076750280794
損失： 0.6066108182836658 	正解率： 0.7863159865219019
損失： 0.5563414643259307 	正解率： 0.8042867839760389
損失： 0.5258141986908625 	正解率： 0.8118682141520029
損失： 0.5076059116619432 	正解率： 0.814301759640584
損失： 0.4945410249913725 	正解率： 0.8172032946461999

Process finished with exit code 0
"""

"""
※shuffleなしの時はまだうまく学習できているみたいです。(?)

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
1
損失： 2.7223920492285356 	正解率： 0.42306252339947586
損失： 2.6847593199385287 	正解率： 0.45039311119430925
損失： 2.6708533726668295 	正解率： 0.45975290153500564
損失： 2.617908932369014 	正解率： 0.46920628977910894
損失： 2.616249207755595 	正解率： 0.4694870834893298
損失： 2.5323187674292904 	正解率： 0.4866154998128042
損失： 2.5075236715653237 	正解率： 0.4942905278921752
損失： 2.4229213022312037 	正解率： 0.5116997379258704
損失： 2.371458794315394 	正解率： 0.520872332459753
損失： 2.3347344915566093 	正解率： 0.528266566828903

Process finished with exit code 0

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
2
損失： 1.4909583781856546 	正解率： 0.5037439161362786
損失： 1.4203652831244948 	正解率： 0.5534444028453762
損失： 1.429448528054548 	正解率： 0.548764507675028
損失： 1.4322288959114815 	正解率： 0.5545675776862599
損失： 1.3572314124622629 	正解率： 0.5811493822538375
損失： 1.3147206579213693 	正解率： 0.5928491201797079
損失： 1.266913748412807 	正解率： 0.6086671658554849
損失： 1.2366017993584006 	正解率： 0.6206476974915762
損失： 1.2165511672449545 	正解率： 0.623736428304006
損失： 1.1992825932610818 	正解率： 0.6300074878322726

Process finished with exit code 0

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
4
損失： 1.1075593902363687 	正解率： 0.5943466866342194
損失： 0.9776756252364044 	正解率： 0.6609883938599775
損失： 0.9120701394510744 	正解率： 0.6876637963309622
損失： 0.9063323329787566 	正解率： 0.6892549606888806
損失： 0.8963599623529954 	正解率： 0.6888805690752527
損失： 0.8651785149372548 	正解率： 0.7054473979782853
損失： 0.9320228486030719 	正解率： 0.6770872332459753
損失： 0.9390514211049942 	正解率： 0.6697865967802321
損失： 0.8891022754250808 	正解率： 0.6924372894047174
損失： 0.8897308390486451 	正解率： 0.6870086110071134

Process finished with exit code 0

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
8
損失： 1.0051976070142614 	正解率： 0.6251403968551105
損失： 0.8299165792529009 	正解率： 0.7073193560464246
損失： 0.7444800289938907 	正解率： 0.7376450767502808
損失： 0.6861092276832167 	正解率： 0.7549606888805691
損失： 0.6444634344445508 	正解率： 0.7687195806813928
損失： 0.6065866806778588 	正解率： 0.7844440284537626
損失： 0.5648266003149266 	正解率： 0.7943654062149008
損失： 0.5194868613815237 	正解率： 0.8140209659303631
損失： 0.5460785033695768 	正解率： 0.8005428678397604
損失： 0.5293837698541782 	正解率： 0.8107450393111194

Process finished with exit code 0
"""

"""
<GPU結果>（文字化けしています）
root@4c53d65ffecc:/workspace/100knock/100knock_chapter9_ver2# CUDA_VISIBLE_DEVICES=0 ./test.sh 83 1
______________1
______ 2.720493222679908        ________ 0.42699363534256835
______ 2.757336755429971        ________ 0.43335829277424187
______ 2.728499494328431        ________ 0.44449644327967053
______ 2.6808617180014584       ________ 0.4559153874953201
______ 2.6695167091913947       ________ 0.4601272931486335
______ 2.6563373875725094       ________ 0.46274803444402846
______ 2.6372473649722457       ________ 0.4647135904155747
______ 2.6371600627787593       ________ 0.4672407338075627
______ 2.629130419481787        ________ 0.4680831149382254
______ 2.5263523247158512       ________ 0.4892362411081992
______________2
______ 1.4934320002738477       ________ 0.5004679895170349
______ 1.394616504937957        ________ 0.5639273680269562
______ 1.4122609249474805       ________ 0.559153874953201
______ 1.3780910707577834       ________ 0.5742231374017222
______ 1.4117102801565462       ________ 0.5615874204417821
______ 1.4193192307967364       ________ 0.5590602770497941
______ 1.3827805291932427       ________ 0.5759078996630476
______ 1.3609475877480488       ________ 0.5789966304754773
______ 1.3608983087544135       ________ 0.5766566828903033
______ 1.3221836216281926       ________ 0.5870460501684762
______________4
______ 1.1311044863148692       ________ 0.5831149382253837
______ 1.0172096155123602       ________ 0.6434855859228753
______ 0.9358492747133299       ________ 0.6768064395357544
______ 0.8899728675229591       ________ 0.6966491950580307
______ 0.8678157797291769       ________ 0.7015162860351928
______ 0.8854550606846944       ________ 0.6940284537626357
______ 0.8629131735442328       ________ 0.7000187195806814
______ 0.8209199339586535       ________ 0.7150879820292025
______ 0.7952598342898632       ________ 0.7275365031823288
______ 0.7501923574385059       ________ 0.7407338075627106
______________8
______ 1.0068239957250342       ________ 0.6248596031448895
______ 0.8241796460850331       ________ 0.7097529015350056
______ 0.7271597919662198       ________ 0.7422313740172221
______ 0.6774994478767326       ________ 0.7626357169599401
______ 0.6125790373126636       ________ 0.7796705353800075
______ 0.614130861109156        ________ 0.7803257207038562
______ 0.5317344710587741       ________ 0.8088730812429802
______ 0.5327630570426047       ________ 0.8051291651067016
______ 0.49535423076661406      ________ 0.8197304380381879
______ 0.5322043993872797       ________ 0.8083114938225384
"""