# coding: utf-8
# xorpytorch.py by kanda.motohiro@gmail.com
# released under The MIT License.
# inspired by
# http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture8.pdf slide 8-9696
import torch
import numpy as np
import matplotlib.pyplot as plt
from util import *
sys.path.append(os.pardir)
import common.functions
"""
XOR 回路を学習する、 pytorch を使った二層のネットワーク。
"""
iters_num = 1000
learning_rate = 0.1

h = 1e-4 # 0.0001
input = np.array(([h, h], [1.0, h], [h, 1.0], [1.0, 1.0]), dtype=np.float32)
xor_output = np.array(([[1.0, h], [h, 1.0], [h, 1.0], [1.0, h]]), dtype=np.float32)


def print_network(L1, L2):
    print("L1 weight")
    print(L1.weight.data)
    print("bias")
    print(L1.bias.data)
    print("L2 weight")
    print(L2.weight.data)
    print("bias")
    print(L2.bias.data)


def predict(x, L1, L2):
    W1 = L1.weight.data.numpy()
    b1 = L1.bias.data.numpy()
    W2 = L2.weight.data.numpy()
    b2 = L2.bias.data.numpy()

    # 転置しなくていいのだっけ。
    a1 = np.dot(x, W1) + b1
    z1 = common.functions.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    return a2


def main():
    L1 = torch.nn.Linear(2, 2)
    # andgate.py に合わせて、バイアスはゼロにしよう。
    L1.bias.data = torch.zeros(2)
    L2 = torch.nn.Linear(2, 2)
    L2.bias.data = torch.zeros(2)
    model = torch.nn.Sequential(L1, torch.nn.Sigmoid(), L2)
    # 二乗平均誤差
    loss_fn = torch.nn.MSELoss()

    # 一番普通の勾配降下法
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loss_list = []
    print("Initial weight and bias")
    print_network(L1, L2)

    for i in range(iters_num):
        for in0, out0 in zip(input, xor_output):
            x = torch.from_numpy(in0)            
            y = torch.from_numpy(out0)
            
            # batch にしないといけないそうな。
            y_pred = model(x.unsqueeze(0))
            loss = loss_fn(y_pred, y.unsqueeze(0))

            model.zero_grad()
            loss.backward()

            # model.parameter を更新してくれる。
            optimizer.step()

            # debug
            train_loss_list.append(loss.item())

        # 最初と最後に、表示する。
        if i == 0 or i == (iters_num - 1) or (i % 100) == 0:
            print("i=%d" % i)
            out = predict(input, L1, L2)
            y_pred = model(torch.from_numpy(input).unsqueeze(0))
            print("pred=" + str(out) + str(y_pred))
            loss = loss_fn(torch.from_numpy(out).unsqueeze(0),
                torch.from_numpy(xor_output).unsqueeze(0))
            print("loss=%f" % loss.item())
    # for iter

    print_network(L1, L2)
    out = predict(input, L1, L2)
    print("pred=" + str(out))
    plot(np.array(train_loss_list), "XOR:losses")

if __name__ == '__main__':
        main()
