from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def linear(input, output_size, scope=None):
    with tf.variable_scope(scope, default_name='linear'):
        shape = [input.get_shape()[-1], output_size]
        matrix = tf.get_variable("matrix", shape, dtype=tf.float32)
        bias = tf.get_variable("bias", [output_size], dtype=tf.float32)
        output = tf.matmul(input, matrix) + bias
        return output


class LegacyRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, reuse=None):
        super(LegacyRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, h, c, scope=None):
        with tf.variable_scope(scope, default_name="RNN_cell",
                               values=[inputs, h]):

            all_inputs = tf.concat([inputs, h], 1)

            new_h = tf.tanh(linear(all_inputs, self._num_units,
                                   scope="hidden"))

        return new_h, new_h, c


class LegacyGRUCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, reuse=None):
        super(LegacyGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, h, c, scope=None):
        with tf.variable_scope(scope, default_name="GRU_cell",
                               values=[inputs, h]):

            all_inputs = tf.concat([inputs, h], 1)

            r = tf.nn.sigmoid(linear(all_inputs, self._num_units,
                                     scope="reset_gate"))
            z = tf.nn.sigmoid(linear(all_inputs, self._num_units,
                                     scope="update_gate"))
            h_ = tf.tanh(linear(tf.concat([r * h, inputs], 1), self._num_units,
                                scope="candidate_"))
            new_h = (1 - z) * h + z * h_

        return new_h, new_h, c


class LegacyLSTMCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_units, reuse=None):
        super(LegacyLSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def __call__(self, inputs, h, c, scope=None):
        with tf.variable_scope(scope, default_name="lstm_cell",
                               values=[inputs, h]):

            all_inputs = tf.concat([inputs, h], 1)

            f = tf.nn.sigmoid(linear(all_inputs, self._num_units,
                                     scope="forget_gate"))
            i = tf.nn.sigmoid(linear(all_inputs, self._num_units,
                                     scope="input_gate"))
            o = tf.nn.sigmoid(linear(all_inputs, self._num_units,
                                     scope="output_gate"))
            c_ = tf.tanh(linear(all_inputs, self._num_units,
                                scope="candidate_"))
            new_c = f * c + i * c_
            new_h = o * tf.tanh(new_c)

        return new_h, new_h, new_c
