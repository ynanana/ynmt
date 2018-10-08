import tensorflow as tf
import rnn_cell


class Attention():
    def __init__(self, num_units, memory):
        self.num_units = num_units
        self.memory = memory

        k_shape = tf.shape(memory)

        key = tf.reshape(memory, [-1, memory.get_shape().as_list()[-1]])
        key = rnn_cell.linear(key, num_units, scope="key_linear")
        self.key = tf.reshape(key, [k_shape[0], k_shape[1], num_units])

    def __call__(self, query, scope=None):
        with tf.variable_scope(scope, default_name="attention"):
            query = rnn_cell.linear(query, self.num_units, scope="query_linear")
            query = tf.expand_dims(query, 1)

            v = tf.get_variable("attention_v", [self.num_units],
                                dtype=tf.float32)
            score = tf.reduce_sum(v * tf.tanh(self.key + query), [2])

            alignments = tf.nn.softmax(score)
            value = tf.reduce_sum(alignments[:, :, None] * self.memory, axis=1)

            return alignments, value
