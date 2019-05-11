# coding: utf-8
# andgate.py by kanda.motohiro@gmail.com
# released under The MIT License.
# based on ch04/train_neuralnet.py at
# https://github.com/oreilly-japan/deep-learning-from-scratch.git
import sys
import os
sys.path.append(os.pardir)
import numpy as np
from one_layer_net import OneLayerNet
from ch04.two_layer_net import TwoLayerNet
import ch04.two_layer_net
from util import *
from common.functions import sigmoid, relu
"""
AND, OR, NAND, XOR 回路を学習する。1, 2 層のネットワーク。
"""
iters_num = 100  # 繰り返しの回数を適宜設定する
learning_rate = 0.1

# Xavier initialization
use_xavier_initialization = True
if use_xavier_initialization:
    weight_init_std = 1/np.sqrt(2)
else:
    weight_init_std = 0.01  # default of TwoLayerNet

# ゼロだと、掛け算してもゼロなので良くないのでないか。
h = 1e-4 # 0.0001

# pytorch で、 Tensor にするときにエラーになるので、 float32 を使う。
input = np.array(([h, h], [1.0, h], [h, 1.0], [1.0, 1.0]), dtype=np.float32)

# AND(0, 0) is 0
and_output = np.array(([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), dtype=np.float32)

# OR(0, 0) is 0
or_output = np.array(([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]), dtype=np.float32)

# NAND(0, 0) is 1
nand_output = np.array(([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]), dtype=np.float32)

# XOR(0, 0) is 0
xor_output = np.array(([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]), dtype=np.float32)


def main():
    if "--xor" in sys.argv:
        train_xor()
    if "--two" in sys.argv:
        # 多めに学習が必要。
        global iters_num
        iters_num *= 10

    for title, output in (("AND", and_output), ("OR", or_output),
            ("NAND", nand_output), ("XOR", xor_output)):
        if "--two" in sys.argv:
            train_gate(input, output, title, mode="two")
        else:
            train_gate(input, output, title)
    sys.exit(0)


def train_xor():
    global iters_num
    iters_num = 1000
    while True:
        ok = train_gate(input, xor_output, "XOR", mode="xor")
        if ok:
            break
    sys.exit(0)


def train_gate(input, output, title="", mode="one"):
    train_loss_list = []
    if mode == "one":
        network = OneLayerNet(input_size=2, output_size=2, weight_init_std=weight_init_std)
        params = ('W1', 'b1')
    else:
        network = TwoLayerNet(input_size=2, hidden_size=2, output_size=2, weight_init_std=weight_init_std)
        params = ('W1', 'b1', 'W2', 'b2')

    network.problem_type = "regression"
    print("Initial weight and bias")
    print(network)

    for i in range(iters_num):
        for x, y in zip(input, output):
            # 勾配の計算
            if mode == "one":
                grad = network.numerical_gradient(x, y)
            else:
                # gradient のコードは、入力が、バッチであることを前提と
                # している。vstack を使って、同じものをバッチにする。
                # 数は、なんでもいいのだが、 2 だと層の数と同じでまぎらわしい
                # ので、 3 にする。
                grad = network.gradient(np.vstack((x, x, x)), np.vstack((y, y, y)))

            # パラメータの更新
            for key in params:
                network.params[key] -= learning_rate * grad[key]

            # debug
            loss = network.loss(x, y)
            train_loss_list.append(loss)

        # 最初と最後に、表示する。
        if i == 0 or i == (iters_num - 1):
            pass
        else:
            continue

        print("%s: i=%d" % (title, i))
        print("loss=" + str(loss))
        a1 = np.dot(input, network.params["W1"]) + network.params["b1"]
        print("a1=\n" + str(a1))
        if mode == "one":
            continue
        if ch04.two_layer_net.use_sigmoid:
            z1 = sigmoid(a1)
        else:
            z1 = relu(a1)
        a2 = np.dot(z1, network.params["W2"]) + network.params["b2"]
        print("a2=\n" + str(a2))

    # いい結果が出ないときはやり直そう。
    if mode == "xor" and loss > 0.1:
        return False

    print(network)
    plot(np.array(train_loss_list), title + ":losses")
    return True

if __name__ == '__main__':
    main()
