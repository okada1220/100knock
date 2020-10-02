# -*- coding: utf-8 -*-

import sys
import os
import torch
from torch import nn, optim
import utils
import models

"""
TensorDataset, DataLoader を使わない場合
（pack_padded_sequence により PackedSequence を使います）
"""

def main():
    # コマンドライン引数から必要な情報を得ます
    category_table = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    if len(sys.argv) < 4:
        print('python 100knock_chapter9_ver2_83(2) [dict_filepath] [train_filepath] [batch_size]')
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

    # RNNモデルを作ります
    input_size = max(word_id_dict.values()) + 1
    embed_size = 300
    hidden_size = 50
    output_size = 4
    model = models.RNN_simple_minibatch_gpu_2(input_size, embed_size, hidden_size, output_size).cuda()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 訓練を行います
    epoch_num = 10

    # バッチサイズごとに訓練データを区切ります（余りは ramain として最後のバッチとして加えます）
    batch_num = int(len(train_title) / batch_size)
    remain_title = train_title[batch_num * batch_size - 1:]
    remain_len_title = torch.tensor([len(title) for title in remain_title])
    remain_title = sorted(remain_title, key=lambda x: x.shape[0], reverse=True)
    remain_title = nn.utils.rnn.pad_sequence(remain_title, batch_first=True)
    remain_len_title , sort_idx = torch.sort(remain_len_title, descending=True)
    remain_category = train_category[batch_num * batch_size - 1:]
    remain_category = torch.gather(remain_category, -1, sort_idx)

    for epoch in range(epoch_num):
        epoch_loss = []
        epoch_accuracy = 0
        for i in range(batch_num):
            optimizer.zero_grad()
            model.reset(batch_size=batch_size)

            # 訓練データからバッチサイズごとに取り出します
            batch_title = train_title[i * batch_size : (i+1) * batch_size]
            len_title = torch.tensor([len(title) for title in batch_title])
            batch_title = sorted(batch_title, key=lambda x: x.shape[0], reverse=True)
            batch_title = nn.utils.rnn.pad_sequence(batch_title, batch_first=True)
            len_title, sort_idx = torch.sort(len_title, descending=True)
            true_category = train_category[i * batch_size : (i+1) * batch_size]
            true_category = torch.gather(true_category, -1, sort_idx)

            # ＧＰＵに送ります
            batch_title = batch_title.cuda()
            len_title = len_title.cuda()
            true_category = true_category.cuda()

            # 予測・損失の計算・勾配の更新を行います
            pred_train_category = model.forward(batch_title, len_title)
            train_loss = loss(pred_train_category, true_category)
            train_loss.backward()
            optimizer.step()
            epoch_loss.append(train_loss.item())
            epoch_accuracy += (true_category == pred_train_category.argmax(1)).sum().item()

        # remain での訓練を行います
        optimizer.zero_grad()
        model.reset(batch_size=len(remain_title))

        remain_title = remain_title.cuda()
        remain_len_title = remain_len_title.cuda()
        remain_category = remain_category.cuda()

        pred_train_category = model.forward(remain_title, remain_len_title)
        train_loss = loss(pred_train_category, remain_category)
        train_loss.backward()
        optimizer.step()
        epoch_loss.append(train_loss.item())
        epoch_accuracy += (remain_category == pred_train_category.argmax(1)).sum().item()

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
損失： 2.6999221325096454 	正解率： 0.4294271808311494
損失： 2.7312245692190373 	正解率： 0.440658929239985
損失： 2.684593405913946 	正解率： 0.4546986147510296
損失： 2.6320360285616986 	正解率： 0.46798951703481845
損失： 2.5745480951054898 	正解率： 0.4801572444777237
損失： 2.527349036475632 	正解率： 0.48904904530138527
損失： 2.491244759201896 	正解率： 0.4968176712841632
損失： 2.449891112372187 	正解率： 0.5058966679146387
損失： 2.4411006409460048 	正解率： 0.5113253463122426
損失： 2.4082060363935978 	正解率： 0.5142268813178584

Process finished with exit code 0

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
2
損失： 1.487328724956742 	正解率： 0.5022463496817672
損失： 1.3973932144970589 	正解率： 0.5642081617371771
損失： 1.4019058240520406 	正解率： 0.5584986896293523
損失： 1.3839723285811063 	正解率： 0.5662673156121303
損失： 1.3765226935234363 	正解率： 0.5728191688506178
損失： 1.3916714290835608 	正解率： 0.5638337701235492
損失： 1.3387186591403837 	正解率： 0.5830213403219768
損失： 1.322668076069732 	正解率： 0.5888244103332085
損失： 1.3162747786122515 	正解率： 0.595469861475103
損失： 1.286244097559494 	正解率： 0.5983713964807188

Process finished with exit code 0

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
4
損失： 1.1030419760071366 	正解率： 0.6012729314863348
損失： 0.9888146103967628 	正解率： 0.6585548483713964
損失： 0.9255030917785895 	正解率： 0.6785847997004867
損失： 0.8716455760103801 	正解率： 0.6972107824784725
損失： 0.8146819101054358 	正解率： 0.7181767128416323
損失： 0.8118918303683997 	正解率： 0.7149943841257956
損失： 0.8030165766629329 	正解率： 0.7184575065518533
損失： 0.7872554394501906 	正解率： 0.7230438038187944
損失： 0.8564732637710194 	正解率： 0.6997379258704605
損失： 0.8740720487978121 	正解率： 0.6911269187570198

Process finished with exit code 0

辞書となるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
訓練データとなるファイルのパスを入力して下さい。
../100knock_chapter6/train.txt
学習時のバッチサイズを入力して下さい。（整数のみ）
8
損失： 1.0162771309608827 	正解率： 0.621115687008611
損失： 0.8919558424034162 	正解率： 0.6818607263197305
損失： 0.7864299877909219 	正解率： 0.721546237364283
損失： 0.739027377488482 	正解率： 0.7390490453013853
損失： 0.679964085617019 	正解率： 0.755335080494197
損失： 0.6135823565094681 	正解率： 0.7769561961812056
損失： 0.5915776147314732 	正解率： 0.7858479970048671
損失： 0.5534695591178059 	正解率： 0.8001684762261325
損失： 0.5179522016962633 	正解率： 0.8139273680269562
損失： 0.5008499698035843 	正解率： 0.814488955447398

Process finished with exit code 0
"""

"""
<GPU結果>（文字化けしています）
root@4c53d65ffecc:/workspace/100knock/100knock_chapter9_ver2# CUDA_VISIBLE_DEVICES=0 ./test.sh 83 2
______________1
______ 2.693902916739798        ________ 0.4285847997004867
______ 2.830363376145057        ________ 0.42334331710969675
______ 2.7679062952937286       ________ 0.44084612504679893
______ 2.742819592977273        ________ 0.44318607263197307
______ 2.700340896155938        ________ 0.4553538000748783
______ 2.6750327534300733       ________ 0.4605952826656683
______ 2.6282744039817563       ________ 0.46873830026207414
______ 2.5948179653776435       ________ 0.4763197304380382
______ 2.5865230339567518       ________ 0.4780980906027705
______ 2.567538027585975        ________ 0.4824035941594908
______________2
______ 1.4748664752176008       ________ 0.5090789966304755
______ 1.4231866231443016       ________ 0.5517596405840509
______ 1.380923866868666        ________ 0.5693560464245601
______ 1.3564790602136858       ________ 0.5790902283788844
______ 1.4864464341958274       ________ 0.5350992137776114
______ 1.4738991575285265       ________ 0.5365031823287159
______ 1.4426512508195548       ________ 0.5535380007487832
______ 1.437202788394376        ________ 0.5519468363908648
______ 1.4221372647096262       ________ 0.559247472856608
______ 1.3785243220773726       ________ 0.5709472107824785
______________4
______ 1.1055929499795427       ________ 0.5975290153500562
______ 0.990106468624296        ________ 0.6539685511044553
______ 0.951307749554158        ________ 0.6678210408086859
______ 0.8837121751070202       ________ 0.6929988768251591
______ 0.847839473439994        ________ 0.7063833770123549
______ 0.8493235033496233       ________ 0.7044178210408086
______ 0.807142004432034        ________ 0.7143391988019469
______ 0.7940700664435942       ________ 0.7246349681767128
______ 0.8060258114487379       ________ 0.7196742792961438
______ 0.7667540707039887       ________ 0.7329651815799326
______________8
______ 1.001741866754022        ________ 0.6280419318607263
______ 0.8358147819554378       ________ 0.7155559715462374
______ 0.7433691630753393       ________ 0.7452265069262448
______ 0.6689056854550085       ________ 0.7675028079371022
______ 0.6377504063368646       ________ 0.7727442905278922
______ 0.586814031356243        ________ 0.7930550355672032
______ 0.5483849372904368       ________ 0.8036315986521901
______ 0.5682439714123657       ________ 0.8010108573567952
______ 0.5137031572274194       ________ 0.8167353051291651
______ 0.4864062750056765       ________ 0.8244103332085362
"""