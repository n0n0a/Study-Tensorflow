import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


def main():
    rng = np.random.RandomState(123)

    d = 2
    N = 10
    mean = 5.0

    x1 = rng.randn(N, d) + np.array([0.0, 0.0])
    x2 = rng.randn(N, d) + np.array([mean, mean])
    # print(x1)
    # print(x2)

    x = np.r_[x1, x2]

    w = np.zeros(d)
    b = 0.0

    def y(val):
        return step(np.dot(w, x[val]) + b)

    def step(val):
        return 1 * (val > 0)

    def t(val):
        return 1 * (val >= N)

    while True:
        classified = True
        for r in range(N * 2):
            dw = (t(r) - y(r)) * x[r]
            db = (t(r) - y(r))
            w += dw
            b += db
            classified *= (db == 0)
        if classified:
            break
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
