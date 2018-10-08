import tensorflow as tf


def linear(input, output_size, scope=None):
    with tf.variable_scope(scope, default_name='linear'):
        shape = [input.get_shape()[-1], output_size]
        matrix = tf.get_variable("matrix", shape, dtype=tf.float32)
        bias = tf.get_variable("bias", [output_size], dtype=tf.float32)
        output = tf.matmul(input, matrix) + bias
        return output


class LSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, reuse=None):
        super(LSTMCell, self).__init__(_reuse=reuse)
        self._num_units = num_units

    def call(self, inputs, state, scope=None):
        with tf.variable_scope(scope, default_name="lstm_cell"):
            if isinstance(inputs, (list, tuple)):
                inputs = tf.concat(inputs, 1)

            (c, h) = state

            all_inputs = tf.concat([inputs, h], 1)
            lstm_matrix = linear(all_inputs, self._num_units*4)
            i, f, o, c_ = tf.split(lstm_matrix, 4, 1)

            new_c = tf.sigmoid(f) * c + tf.sigmoid(i) * tf.tanh(c_)
            new_h = tf.sigmoid(o) * tf.tanh(c)

            new_state = (new_c, new_h)

        return new_h, new_state

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units


class MultiRNNCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cells):
        super(MultiRNNCell, self).__init__()
        self.cells = cells

    def __call__(self, inputs, state, context=None):
        cur_inp = inputs
        new_states = []

        for i, cell in enumerate(self.cells):
            with tf.variable_scope("cell_%d" % i):
                if context is not None:
                    cur_inp = [cur_inp, context]
                cur_state = state[i]
                cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)

        return cur_inp, tuple(new_states)

    @property
    def state_size(self):
        return tuple((cell.state_size for cell in self.cells))

    @property
    def output_size(self):
        return self.cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        return tuple(cell.zero_state(batch_size, dtype) for cell in self.cells)
