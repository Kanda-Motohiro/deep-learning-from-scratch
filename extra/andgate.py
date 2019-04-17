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
"""
AND 回路を学習する。1/2 層のネットワーク。
"""
iters_num = 100  # 繰り返しの回数を適宜設定する
learning_rate = 0.1

# ゼロだと、掛け算してもゼロなので良くないのでないか。
h = 1e-4 # 0.0001
input = np.array(([h, h], [1.0, h], [h, 1.0], [1.0, 1.0]))

# AND(0, 0) is 0
and_output = np.array(([0.0, 0.0, 0.0, 1.0]))

# OR(0, 0) is 0
or_output = np.array(([0.0, 1.0, 1.0, 1.0]))

# NAND(0, 0) is 1
nand_output = np.array(([1.0, 1.0, 1.0, 0.0]))

# XOR(0, 0) is 0
xor_output = np.array(([0.0, 1.0, 1.0, 0.0]))


def main():
    for title, outputs in (("AND", and_output), ("OR", or_output),
            ("NAND", nand_output), ("XOR", xor_output)):
        print(title)
        if "--two" in sys.argv:
            global iters_num
            iters_num *= 10
            train_gate(input, outputs, title, two=True)
        else:
            train_gate(input, outputs, title)
    sys.exit(0)


def train_gate(inputs, outputs, title="", two=False):
    train_loss_list = []
    if two:
        network = TwoLayerNet(input_size=2, hidden_size=2, output_size=1)
        params = ('W1', 'b1', 'W2', 'b2')
    else:
        network = OneLayerNet(input_size=2, output_size=1)
        params = ('W1', 'b1')

    print(network)

    for i in range(iters_num):
        for j in range(4):
            input = inputs[j]
            output = outputs[j]
            # 勾配の計算
            grad = network.numerical_gradient(input, output)
            #grad = network.gradient(x_batch, t_batch)
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

            if j == 0:
                print("i=%d" % i)
            y = network.predict(input)
            # 見やすいように 1/0 で表示する。 
            print("input=" + str([round(x) for x in input]))
            print("predict=" + str(y[0]))
            print("output=" + str(output))
            print("loss=" + str(loss))
        # for inputs

    print(title)
    print(network)
    plot(np.array(train_loss_list), title + ":losses")
    if not two:
        plotOneLayerNetwork(network, title)
    return


if __name__ == '__main__':
    main()
