import tensorflow as tf
import numpy as np
import reader
import argparse
import model


class Config(object):
    learning_rate = 0.1
    num_layers = 2
    num_steps = 35
    hidden_size = 200
    epoch = 10000
    keep_prob = 0.5
    batch_size = 20
    vocab_size = 10000
    cell_type = 'gru'
    clip = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/',
                        help='Where the data is stored')
    parser.add_argument('--save_path', default='model.ckpt',
                        help='Model output directory')
    parser.add_argument('--cell_type', default='gru',
                        help='the type of cell used')
    return parser.parse_args()


def ptb_input(data, config):
    data_len = len(data)
    data = np.array(data)
    n = data_len // config.num_steps
    x = data[0: (n - 1) * config.num_steps].reshape(((n-1),
                                                    config.num_steps))
    y = data[1: (n - 1) * config.num_steps + 1].reshape(((n-1),
                                                        config.num_steps))
    return x, y


def main(args):
    config = Config()

    config.cell_type = args.cell_type

    train_data, valid_data, test_data, _ = \
        reader.PtbReader(args.data_path).ptb_data()
    print 'loading data finish'

    ptb_reader = reader.PtbReader(args.data_path)
    train_data, valid_data, test_data, vocabulary = ptb_reader.ptb_data()
    train_x, train_y = ptb_input(train_data, config)
    valid_x, valid_y = ptb_input(valid_data, config)
    test_x, test_y = ptb_input(test_data, config)

    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope('l_m', reuse=tf.AUTO_REUSE,
                           initializer=initializer):
        train_model = model.PtbModel(True, config)

    with tf.variable_scope('l_m', reuse=True, initializer=initializer):
        eval_model = model.PtbModel(False, config)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    for step in range(config.epoch):
        loss = 0.0
        for j in range(0, len(train_x) - 1, config.batch_size):
            feed_dict = {}
            mini_batch_x = train_x[j: j + config.batch_size]
            mini_batch_y = train_y[j: j + config.batch_size]
            feed_dict[train_model.xs] = mini_batch_x
            feed_dict[train_model.ys] = mini_batch_y

            sess.run(train_model.train_op, feed_dict=feed_dict)
            print j, ':', sess.run(tf.exp(train_model.cost),
                                   feed_dict=feed_dict)

        """
            loss += train_model.cost
        loss = tf.exp(loss / len(train_x) * config.batch_size)
        print sess.run(loss, feed_dict=feed_dict)
        """

        valid_perplexity = sess.run(tf.exp(eval_model.cost), feed_dict={
                                    eval_model.xs: valid_x,
                                    eval_model.ys: valid_y})
        print step, 'valid_perplexity:', valid_perplexity
        config.learning_rate *= 0.5

    test_perplexity = sess.run(tf.pow(2.7, eval_model.cost), feed_dict={
        eval_model.xs: test_x,
        eval_model.ys: test_y})
    print step, 'test_perplexity:', test_perplexity


if __name__ == "__main__":
    main(parse_args())
