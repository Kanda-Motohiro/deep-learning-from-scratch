# coding: utf-8
# xortensorflow.py by kanda.motohiro@gmail.com
# released under The MIT License.
# inspired by
# http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture8.pdf slide 8-6060
# and https://www.tensorflow.org/tutorials/keras/basic_classification
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

h = 1e-4 # 0.0001
input = np.array(([h, h], [1.0, h], [h, 1.0], [1.0, 1.0]))
xor_output = np.array(([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]))

def print_network(L1, L2):
    print("L1 weight, bias")
    print(L1.get_weights())
    print("L2 weight, bias")
    print(L2.get_weights())


def main():
    model = keras.Sequential()
    L1 = keras.layers.Dense(2, activation=tf.nn.sigmoid,
            kernel_initializer='glorot_normal', bias_initializer='zeros')
    L2 = keras.layers.Dense(2)
    model.add(L1)
    model.add(L2)
    model.compile(optimizer='adam', loss="mean_squared_error")

    # 学習。
    history = model.fit(input, xor_output, verbose=0, epochs=1000)

    # 重みを見る。
    print_network(L1, L2)

    # 結果を見る。
    y_pred = model.predict(input)
    print("pred=" + str(y_pred))


if __name__ == '__main__':
        main()
