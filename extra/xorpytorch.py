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
from andgate import learning_rate, input, and_output, or_output, nand_output, xor_output
"""
AND, OR, NAND, XOR 回路を学習する、 pytorch を使った二層のネットワーク。
"""
iters_num = 100


def print_network(L1, L2):
    print("L1 weight")
    print(L1.weight.data)
    print("bias")
    print(L1.bias.data)
    print("L2 weight")
    print(L2.weight.data)
    print("bias")
    print(L2.bias.data)


def main():
    if "--xor" in sys.argv:
        train_gate(input, xor_output, "XOR")
        sys.exit(0)

    for title, output in (("AND", and_output), ("OR", or_output),
        ("NAND", nand_output), ("XOR", xor_output)):
        train_gate(input, output, title)
    sys.exit(0)


def train_gate(input, output, title=""):
    L1 = torch.nn.Linear(2, 2)
    # andgate.py に合わせて、バイアスはゼロにしよう。
    L1.bias.data = torch.zeros(2)
    L2 = torch.nn.Linear(2, 2)
    L2.bias.data = torch.zeros(2)
    model = torch.nn.Sequential(L1, torch.nn.Sigmoid(), L2)

    # 二乗平均誤差
    loss_fn = torch.nn.MSELoss()

    # SGD の時は全然、学習しなかったのだが、 Adam にしたら良くなった。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_list = []
    print("Initial weight and bias")
    print_network(L1, L2)

    for i in range(iters_num):
        for in0, out0 in zip(input, output):
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
        if i == 0 or i == (iters_num - 1):
            pass
        else:
            continue

        print("%s: i=%d" % (title, i))

        y_pred = model(torch.from_numpy(input).unsqueeze(0))

        loss = loss_fn(y_pred,
            torch.from_numpy(output).unsqueeze(0))
        print("loss=%f" % loss.item())
        print("pred=" + str(y_pred))
    # for iter

    print_network(L1, L2)
    plot(np.array(train_loss_list), title + ":losses")

if __name__ == '__main__':
        main()
