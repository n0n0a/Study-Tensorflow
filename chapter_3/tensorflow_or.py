import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


def main():
    d = 2

    tf.set_random_seed(0)
    w = tf.Variable(tf.zeros(d))
    b = tf.Variable(tf.zeros(1))

    x = tf.placeholder(tf.float32, shape=[None, 2])
    t = tf.placeholder(tf.float32, shpae=[None, 1])
    y = tf.nn.sigmoid(tf.matmul(w, x)+b)

    cross_entropy = -tf.reduce_sum(t*tf.log(y)+(1-t)*tf.log(1-y))
    train_step = tf.train.GradientDescentOptimize(0.1).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.to_fload(tf.greater(y, 0.5),t))

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[1]])

    init = tf.global_varialbes_initializer()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(200):
        sess.run(train_step, feed_dict={
            x:X,
            t:Y
        })

    classified = correct_prediction.eval(session=sess, feed_dict={
        x:X,
        t:Y
    })

    prob = y.eval(session=sess, feed_dict={
        x:X,
        t:Y
    })

    print('classified: ')
    print(classified)
    print('probablity: ')
    print(prob)

    print("Hello")

if __name__ == '__main__':
    main()
