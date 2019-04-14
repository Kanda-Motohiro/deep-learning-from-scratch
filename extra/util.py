# coding: utf-8
# util.py by kanda.motohiro@gmail.com
# released under The MIT License.
# based on ch04/train_neuralnet.py at
# https://github.com/oreilly-japan/deep-learning-from-scratch.git
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
"""
部品。主に、画像とグラフを表示する。
"""


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def plot2DMatrix(matrix, title=None):
    "二次元行列の散布図を描く。"
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    ax = Axes3D(fig)
    x_size = matrix.shape[0]
    y_size = matrix.shape[1]
    x = np.arange(x_size)
    y = np.arange(y_size)
    X, Y = np.meshgrid(x, y)
    ax.scatter3D(np.ravel(X), np.ravel(Y), np.ravel(matrix))
    plt.show()


def quiver2DMatrix(matrix, title=None):
    "二次元行列を Y 軸方向の矢印の大きさで表示する。"
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    plt.quiver(np.zeros(matrix.shape), matrix)
    plt.draw()
    plt.show()


def plotArray(array, title=None):
    "配列の散布図を描く。"
    if title is not None:
        plt.title(title)
    x = np.arange(array.size)
    plt.scatter(x, array)
    plt.show()


def plot(obj, title=None):
    if obj.ndim == 1:
        plotArray(obj, title)
    else:
        plot2DMatrix(obj, title)


def plotNetwork(network, iters):
    for key in ('W1', 'b1', 'W2', 'b2'):
        if key not in network.params:
            continue
        print(key)
        param = network.params[key]
        plot(param, key + " iters=%d" % iters)


def plotOneLayerNetwork(network, title):
    # W1[0]x + W1[1]y + b1 = 0 のグラフを引く。
    if title is not None:
        plt.title(title)
    x = np.arange(0.0, 1.1, 0.1)
    b1 = network.params["b1"]
    W11 = network.params["W1"][0]
    W12 = network.params["W1"][1]
    y = (-b1 - W11 * x)/W12
    plt.plot(x, y)
    plt.grid(True)
    #plt.xlim((0, 1))
    #plt.ylim((0, 1))
    plt.show()
