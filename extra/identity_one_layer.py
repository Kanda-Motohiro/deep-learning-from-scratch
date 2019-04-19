# coding: utf-8
# identity_one_layer.py by kanda.motohiro@gmail.com
# released under The MIT License.
# based on ch04/train_neuralnet.py at
# https://github.com/oreilly-japan/deep-learning-from-scratch.git
import sys
import numpy as np
from one_layer_net import OneLayerNet
from util import *
"""
四次元のベクトルを恒等変換する行列を、一層ネットワークで学習する。
1   1 0 0 0   1
0 * 0 1 0 0 = 0
0   0 0 1 0   0
0   0 0 0 1   0
"""
network = OneLayerNet(input_size=4, output_size=4)

iters_num = 50  # 繰り返しの回数を適宜設定する
learning_rate = 0.1

train_loss_list = []

h = 1e-4 # 0.0001
imgs = np.array(([1.0,h,h,h], [h,1.0,h,h], [h,h,1.0,h], [h,h,h,1.0]))

for i in range(iters_num):
    debug_gradient_total =np.zeros((4, 4))
    if (i % 10) == 0:
        print("i=%d" % i)
    for img in imgs:
        label = img # 入力と出力は等しい。

        # 勾配の計算
        grad = network.numerical_gradient(img, label)
        debug_gradient_total += grad['W1']
        #grad = network.gradient(x_batch, t_batch)
        
        # パラメータの更新
        for key in ('W1', 'b1'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(img, label)
        train_loss_list.append(loss)
        # 時々、表示する。
        if (i % 10) != 0:
            continue
        input, a1 = network.explain(img)
        print("input=" + str([round(x) for x in input]))
        print("a1=" + str(a1))
    # for imgs

    # 勾配を表示する。
    if i == 0:
        quiver2DMatrix(debug_gradient_total * -1, 'W1 gradient')
    
print("W1")
print(network.params["W1"])
plot(np.array(train_loss_list), "losses")
plotNetwork(network, iters_num)
sys.exit(0)
