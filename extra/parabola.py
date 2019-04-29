# coding: utf-8
# parabola.py by kanda.motohiro@gmail.com
# released under The MIT License.
# based on ch04/train_neuralnet.py at
# https://github.com/oreilly-japan/deep-learning-from-scratch.git
import sys
import os
sys.path.append(os.pardir)
import numpy as np
from ch04.two_layer_net import TwoLayerNet
from util import *
from common.functions import sigmoid
"""
放物線を回帰する。
"""
iters_num = 10  # 繰り返しの回数を適宜設定する
learning_rate = 0.1

input = np.arange(0, 1.0, 0.1)
output = input * input
input2 = np.arange(0, 10.0, 1)
output2 = np.sin(input2)

def main():
    train(input, output)
    train(input2, output2)
    sys.exit(0)


def train(input, output):
    train_loss_list = []
    network = TwoLayerNet(input_size=10, hidden_size=10, output_size=10)
    params = ('W1', 'b1', 'W2', 'b2')

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

        print("i=%d" % i)
        print("loss=" + str(loss))
        a1 = np.dot(input, network.params["W1"]) + network.params["b1"]
        print("a1=\n" + str(a1))
        a2 = np.dot(sigmoid(a1), network.params["W2"]) + network.params["b2"]
        print("a2=\n" + str(a2))

    print(network)
    plt.plot(input, a2)
    plt.plot(input, output)
    plt.show()
    plot(np.array(train_loss_list), "losses")
    return

if __name__ == '__main__':
    main()
