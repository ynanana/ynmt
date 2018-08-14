import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cPickle


class Net:
    def __init__(self, size):
        self.w1 = tf.Variable(tf.random_normal([size[0], size[1]]))
        self.b1 = tf.Variable(tf.zeros([size[1]]) + 0.1)
        self.w2 = tf.Variable(tf.random_normal([size[1], size[2]]))
        self.b2 = tf.Variable(tf.zeros([ size[2]]) + 0.1)

    def forward(self, inputs):
        l1_out = tf.nn.sigmoid(tf.matmul(inputs, self.w1) + self.b1)
        l2_out = tf.nn.softmax(tf.matmul(l1_out, self.w2) + self.b2)
        return l2_out

    def evaluation(self, x, y):
        y_ = self.forward(x)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        return accuracy


def main():
    fd = open("mnist.pkl", "rb")
    data = cPickle.load(fd)
    fd.close()

    train_data_x, train_data_y = data[0]
    valid_data_x, valid_data_y = data[1]
    test_data_x, test_data_y = data[2]

    epochs = 50
    n = len(train_data_x)
    mini_batch_size = 128

    train_data_y = np.eye(len(train_data_y), 10)[train_data_y]
    valid_data_y = np.eye(len(valid_data_y), 10)[valid_data_y]
    test_data_y = np.eye(len(test_data_y), 10)[test_data_y]

    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    net = Net([784, 20, 10])
    y_ = net.forward(xs)
    cross_entropy = -tf.reduce_sum(ys * tf.log(y_))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    accuracy = net.evaluation(xs, ys)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(epochs):
        for j in range(0, n, mini_batch_size):
            mini_batch_x = train_data_x[j: j + mini_batch_size]
            mini_batch_y = train_data_y[j: j + mini_batch_size]
            sess.run(train_step, feed_dict={xs: mini_batch_x, ys: mini_batch_y})
            """
            if j % 50 == 0:
                print j, ':', sess.run(cross_entropy, feed_dict={
                    xs: mini_batch_x, ys: mini_batch_y})
                    """
        if j < n-1:
            mini_batch_x = train_data_x[j:]
            mini_batch_y = train_data_y[j:]
            sess.run(train_step, feed_dict={xs: mini_batch_x, ys: mini_batch_y})

        print "accuracy of valid_data is: ", sess.run(accuracy, feed_dict={
            xs: valid_data_x, ys: valid_data_y})
        print "accuracy of test_data is: ", sess.run(accuracy, feed_dict={
            xs: test_data_x, ys: test_data_y})


if __name__ == "__main__":
    main()
