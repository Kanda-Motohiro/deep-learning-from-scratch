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

    for title, outputs in (("AND", and_output), ("OR", or_output),
            ("NAND", nand_output), ("XOR", xor_output)):
        print(title)
        if "--two" in sys.argv:
            train_gate(input, outputs, title, two=True)
        else:
            train_gate(input, outputs, title)
    sys.exit(0)


def train_xor():
    global iters_num
    iters_num = 10000
    train_gate(input, xor_output, "XOR", two=True)
    sys.exit(0)


def train_gate(inputs, outputs, title="", two=False):
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
            print("predict=" + str(y))
            print("output=" + str(output))
            print("loss=" + str(loss))
            if two:
                x, a1, z1, a2 = network.explain(input)
                print(x, a1, z1, a2)
        # for inputs

    print(title)
    print(network)
    plot(np.array(train_loss_list), title + ":losses")
    #if not two:
    #    plotOneLayerNetwork(network, title)
    return

"""
train_gate 先頭で、正解を与えたのに、ずれていくのはなぜ。

        network.params["W1"] = np.array(([[1.0, 1.0], [1.0,1.0]]))
        network.params["b1"] = np.array(([0.0, -1.0]))
        network.params["W2"] = np.array(([1.0, -2.0]))
        network.params["b2"] = np.array(([0.0]))

input=[0.0, 0.0]
predict=[-0.03207486]
output=0.0
loss=0.0005143984216133592
[0.0001 0.0001] [ 0.00114779 -1.00129091] [0.50028695 0.26868769] [-0.03207486]
input, W1*input+b1, sigmoid, W2*a2+b2, output と並んでいるわけだが、

sigmoid をかけたことで、行列計算の結果と変わってしまったわけ。
"""

if __name__ == '__main__':
    main()
