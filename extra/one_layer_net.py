# coding: utf-8
# one_layer_net.py by kanda.motohiro@gmail.com
# released under The MIT License.
# based on ch04/two_layer_net.py at
# https://github.com/oreilly-japan/deep-learning-from-scratch.git
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common.functions import *
from common.gradient import numerical_gradient
"""
１層のネットワーク。
"""

class OneLayerNet:
    def __init__(self, input_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1'] = np.zeros(output_size)

    def predict(self, x):
        W1 = self.params['W1']
        b1 = self.params['b1']

        a1 = np.dot(x, W1) + b1
        # 本で扱っている問題は、分類。今、やりたいのは、回帰。なので、a1 を返す。
        #y = softmax(a1)
        y = a1
        return y

    def explain(self, x):
        """デバッグのため、入力 x が各層を伝わっていく途中を返す。
        returns input, hidden-layer1, output
        """
        W1 = self.params['W1']
        b1 = self.params['b1']

        a1 = np.dot(x, W1) + b1
        return x, a1

    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)

        # これも、回帰だから、こっちを使う。
        #return cross_entropy_error(y, t)
        return mean_squared_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])

        return grads

    def __repr__(self):
        out = "W1\n" + str(self.params["W1"]) + "\n"
        out += "b1\n" + str(self.params["b1"])
        return out
