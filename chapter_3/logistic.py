import numpy as np
import math
import random
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


def main():
    rng = np.random.RandomState(123)

    d = 2
    N = 100
    mean = 5.0
    epoch = 10

    x1 = rng.randn(N, d) + np.array([0.0, 0.0])
    x2 = rng.randn(N, d) + np.array([mean, mean])
    # print(x1)
    # print(x2)

    x = np.r_[x1, x2]

    w = np.zeros(d)
    b = 0.0

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def y(val):
        return sigmoid(np.dot(w, x[val]) + b)

    def step(val):
        return 1 * (val > 0)

    def t(val):
        return 1 * (val >= N)

    total_dw = 0
    total_db = 0
    gamma = 0.5
    div = int(math.sqrt(N))

    rs = list(range(2*N))
    for _ in range(epoch):
        random.shuffle(rs)
        for r in rs:
            total_dw += (t(r) - y(r)) * x[r]
            total_db += (t(r) - y(r))
            if(r+1) % div:
                w += gamma*total_dw
                b += gamma*total_db
                total_dw=0
                total_db=0
    print(w, b)

    xs = np.linspace(-1,7,100)
    ys = -w[0]/w[1]*xs-b/w[1]
    plt.plot(xs, ys)
    plt.scatter(x[:N, 0], x[:N, 1], marker='.')
    plt.scatter(x[N:, 0], x[N:, 1], marker='^')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
