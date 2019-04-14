# coding: utf-8
# identity_two_layer.py by kanda.motohiro@gmail.com
# released under The MIT License.
# based on ch04/train_neuralnet.py at
# https://github.com/oreilly-japan/deep-learning-from-scratch.git
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from ch04.two_layer_net import TwoLayerNet
from util import *
"""
二層ネットワークで、恒等変換を学習する。
"""
network = TwoLayerNet(input_size=4, hidden_size=4, output_size=4)

iters_num = 250  # 繰り返しの回数を適宜設定する
learning_rate = 0.1

train_loss_list = []

h = 1e-4 # 0.0001
imgs = np.array(([1.0,h,h,h], [h,1.0,h,h], [h,h,1.0,h], [h,h,h,1.0]))

for i in range(iters_num):
    if (i % 100) == 0:
        print("i=%d" % i)
    for img in imgs:
        label = img # 入力と出力は等しい。

        # 勾配の計算
        grad = network.numerical_gradient(img, label)
        #grad = network.gradient(x_batch, t_batch)
        
        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(img, label)
        train_loss_list.append(loss)
        # 時々、表示する。
        if (i % 100) != 0:
            continue
        input, a1, sigmoid, a2, softmax = network.explain(img)
        print("input=" + str([round(x) for x in input]))
        print("a1=" + str(a1))
        print("sigmoid=" + str(sigmoid))
        print("a2=" + str(a2))
        print("softmax=" + str(softmax))
    # for imgs
    
plot(np.array(train_loss_list), "losses")
plotNetwork(network, iters_num)
sys.exit(0)
