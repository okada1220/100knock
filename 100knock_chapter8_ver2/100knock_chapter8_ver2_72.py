import os
import torch
from torch import nn
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
            parameters = singlenet_model.parameters()
            for parameter in parameters:
                print(parameter)
                print(parameter.grad)

            pred_y1_vector = singlenet_model.forward(x_matrix[:1])
            print('y1:')
            print(pred_y1_vector)
            l1_loss = loss(pred_y1_vector, y_vector[:1])
            print('l1_loss:')
            print(l1_loss)
            l1_loss.backward()
            print('l1_grad:')
            parameters = singlenet_model.parameters()
            for parameter in parameters:
                print(parameter.grad)

            pred_y_matrix = singlenet_model.forward(x_matrix[:4])
            print('Y:')
            print(pred_y_matrix)
            l_loss = loss(pred_y_matrix, y_vector[:4])
            print('L_loss:')
            print(l_loss)
            l_loss.backward()
            print('L_grad:')
            parameters = singlenet_model.parameters()
            for parameter in parameters:
                print(parameter.grad)
        else:
            print('対象ファイルが存在しません。\n')
        print('次の対象ファイルへのパスを入力して下さい。（終了：Enter）')  # 連続して処理を行います
        filepath = input()

if __name__ == '__main__':
    main()

"""
<実行結果>

対象ファイルへのパスを入力して下さい。
train_vectorizer.npz
Parameter containing:
tensor([[ 0.0215,  0.0348, -0.0418,  ..., -0.0308,  0.0427, -0.0239],
        [-0.0216,  0.0069,  0.0562,  ..., -0.0531, -0.0232,  0.0467],
        [ 0.0074,  0.0056,  0.0546,  ...,  0.0059, -0.0031, -0.0136],
        [-0.0495, -0.0364, -0.0497,  ..., -0.0495, -0.0543,  0.0121]],
       requires_grad=True)
None
y1:
tensor([[ 0.0269,  0.0479, -0.0033,  0.0329]], grad_fn=<MmBackward>)
l1_loss:
tensor(1.4158, grad_fn=<NllLossBackward>)
l1_grad:
tensor([[ 0.0165,  0.0098, -0.0101,  ..., -0.0191, -0.0019,  0.0055],
        [ 0.0168,  0.0101, -0.0104,  ..., -0.0195, -0.0020,  0.0056],
        [-0.0499, -0.0298,  0.0307,  ...,  0.0579,  0.0059, -0.0166],
        [ 0.0166,  0.0099, -0.0102,  ..., -0.0192, -0.0020,  0.0055]])
Y:
tensor([[ 0.0269,  0.0479, -0.0033,  0.0329],
        [ 0.0234, -0.0040,  0.0654,  0.0928],
        [ 0.0496,  0.0270, -0.0305,  0.0790],
        [ 0.0340, -0.0069,  0.0147,  0.0747]], grad_fn=<MmBackward>)
L_loss:
tensor(1.4137, grad_fn=<NllLossBackward>)
L_grad:
tensor([[ 0.0384,  0.0159,  0.0200,  ..., -0.0664, -0.0320,  0.0205],
        [ 0.0179,  0.0090, -0.0236,  ..., -0.0150,  0.0122,  0.0050],
        [-0.0735, -0.0335,  0.0280,  ...,  0.0949,  0.0063, -0.0302],
        [ 0.0172,  0.0086, -0.0244,  ..., -0.0136,  0.0135,  0.0047]])
次の対象ファイルへのパスを入力して下さい。（終了：Enter）
valid_vectorizer.npz
Parameter containing:
tensor([[-0.0003,  0.0300, -0.0283,  ..., -0.0186,  0.0529, -0.0088],
        [ 0.0281,  0.0225, -0.0496,  ..., -0.0501,  0.0491,  0.0375],
        [-0.0045, -0.0013,  0.0197,  ...,  0.0372, -0.0098, -0.0169],
        [ 0.0487, -0.0296, -0.0357,  ...,  0.0504,  0.0027, -0.0437]],
       requires_grad=True)
None
y1:
tensor([[-0.0269, -0.0341,  0.0762, -0.0532]], grad_fn=<MmBackward>)
l1_loss:
tensor(1.4050, grad_fn=<NllLossBackward>)
l1_grad:
tensor([[-0.0507, -0.0351,  0.0049,  ...,  0.0199, -0.0339,  0.0655],
        [ 0.0164,  0.0113, -0.0016,  ..., -0.0064,  0.0109, -0.0212],
        [ 0.0183,  0.0127, -0.0018,  ..., -0.0072,  0.0122, -0.0236],
        [ 0.0161,  0.0111, -0.0015,  ..., -0.0063,  0.0107, -0.0208]])
Y:
tensor([[-0.0269, -0.0341,  0.0762, -0.0532],
        [ 0.0390,  0.0257,  0.0022, -0.0293],
        [-0.1562, -0.0425,  0.0697,  0.0279],
        [-0.0601, -0.0227,  0.1110, -0.0455]], grad_fn=<MmBackward>)
L_loss:
tensor(1.4410, grad_fn=<NllLossBackward>)
L_grad:
tensor([[-0.0882, -0.0405,  0.0180,  ...,  0.0112, -0.0465,  0.1204],
        [ 0.0374,  0.0181, -0.0032,  ..., -0.0114,  0.0215, -0.0350],
        [ 0.0134,  0.0049, -0.0110,  ...,  0.0112,  0.0038, -0.0500],
        [ 0.0374,  0.0175, -0.0038,  ..., -0.0110,  0.0212, -0.0353]])
次の対象ファイルへのパスを入力して下さい。（終了：Enter）
test_vectorizer.npz
Parameter containing:
tensor([[-0.0044, -0.0283,  0.0018,  ..., -0.0535, -0.0151,  0.0235],
        [ 0.0261,  0.0455, -0.0503,  ..., -0.0273, -0.0268,  0.0390],
        [ 0.0517,  0.0311, -0.0287,  ...,  0.0348, -0.0076,  0.0088],
        [-0.0018,  0.0445, -0.0138,  ...,  0.0072,  0.0033, -0.0027]],
       requires_grad=True)
None
y1:
tensor([[-0.0370, -0.1163,  0.1101, -0.0181]], grad_fn=<MmBackward>)
l1_loss:
tensor(1.4113, grad_fn=<NllLossBackward>)
l1_grad:
tensor([[-0.0028, -0.0356, -0.0578,  ...,  0.0227, -0.0050,  0.0070],
        [ 0.0008,  0.0106,  0.0172,  ..., -0.0068,  0.0015, -0.0021],
        [ 0.0010,  0.0133,  0.0216,  ..., -0.0085,  0.0019, -0.0026],
        [ 0.0009,  0.0117,  0.0190,  ..., -0.0074,  0.0016, -0.0023]])
Y:
tensor([[-0.0370, -0.1163,  0.1101, -0.0181],
        [-0.0235, -0.0654,  0.0811,  0.0236],
        [-0.0200, -0.0423,  0.0896, -0.0144],
        [-0.0526, -0.0443,  0.0983,  0.0470]], grad_fn=<MmBackward>)
L_loss:
tensor(1.4212, grad_fn=<NllLossBackward>)
L_grad:
tensor([[ 0.0017, -0.0436, -0.0929,  ...,  0.0227,  0.0041,  0.0058],
        [ 0.0074,  0.0160,  0.0269,  ..., -0.0073,  0.0026, -0.0082],
        [ 0.0086,  0.0197,  0.0333,  ..., -0.0092,  0.0031, -0.0096],
        [-0.0177,  0.0079,  0.0327,  ..., -0.0061, -0.0098,  0.0120]])
次の対象ファイルへのパスを入力して下さい。（終了：Enter）


Process finished with exit code 0
"""