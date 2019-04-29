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
from util import *
from common.functions import sigmoid
"""
AND 回路を学習する。1, 2 層のネットワーク。
"""
iters_num = 100  # 繰り返しの回数を適宜設定する
learning_rate = 0.1

# ゼロだと、掛け算してもゼロなので良くないのでないか。
h = 1e-4 # 0.0001
input = np.array(([h, h], [1.0, h], [h, 1.0], [1.0, 1.0]))

# AND(0, 0) is 0
and_output = np.array(([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

# OR(0, 0) is 0
or_output = np.array(([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))

# NAND(0, 0) is 1
nand_output = np.array(([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]))

# XOR(0, 0) is 0
xor_output = np.array(([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]))


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
            train_gate(input, output, title, two=True)
        else:
            train_gate(input, output, title)
    sys.exit(0)


def train_xor():
    global iters_num
    iters_num = 10000
    train_gate(input, xor_output, "XOR", two=True)
    sys.exit(0)


def train_gate(input, output, title="", two=False):
    train_loss_list = []
    if two:
        network = TwoLayerNet(input_size=2, hidden_size=2, output_size=2)
        params = ('W1', 'b1', 'W2', 'b2')
    else:
        network = OneLayerNet(input_size=2, output_size=2)
        params = ('W1', 'b1')

    network.problem_type = "regression"
    print("Initial weight and bias")
    print(network)

    for i in range(iters_num):
        # 勾配の計算
        grad = network.numerical_gradient(input, output)
        #grad = network.gradient(input, output)
        #print(grad)

        # パラメータの更新
        for key in params:
            network.params[key] -= learning_rate * grad[key]

        # debug
        loss = network.loss(input, output)
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
        if not two:
            continue
        a2 = np.dot(sigmoid(a1), network.params["W2"]) + network.params["b2"]
        print("a2=\n" + str(a2))

    print(network)
    plot(np.array(train_loss_list), title + ":losses")
    #if not two:
    #    plotOneLayerNetwork(network, title)
    return

if __name__ == '__main__':
    main()
