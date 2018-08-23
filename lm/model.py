import tensorflow as tf
import numpy as np
import rnn_cell


class PtbModel:
    def __init__(self, is_training, config):
        self.xs = tf.placeholder(tf.int32, [None, config.num_steps])
        self.ys = tf.placeholder(tf.int32, [None, config.num_steps])

        embedding = tf.get_variable("embedding", [config.vocab_size,
                                                  config.hidden_size],
                                    dtype=tf.float32)

        if config.cell_type == 'rnn':
            print 'rnn'
            cell = rnn_cell.LegacyRNNCell(config.hidden_size)
        elif config.cell_type == 'lstm':
            print 'lstm'
            cell = rnn_cell.LegacyLSTMCell(config.hidden_size)
        else:
            print 'gru'
            cell = rnn_cell.LegacyGRUCell(config.hidden_size)

        inputs = tf.nn.embedding_lookup(embedding, self.xs)

        if is_training:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        init_h = tf.zeros([tf.shape(self.xs)[0], config.hidden_size],
                          tf.float32)
        init_c = tf.zeros([tf.shape(self.xs)[0], config.hidden_size],
                          tf.float32)

        input_ta = tf.TensorArray(tf.float32, config.num_steps,
                                  tensor_array_name='input_array')
        output_ta = tf.TensorArray(tf.float32, config.num_steps,
                                   tensor_array_name='output_array')
        input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

        def loop_func(t, out_ta, h, c):
            inp_t = input_ta.read(t)
            cell_output, new_h, new_c = cell(inp_t, h, c)
            out_ta = out_ta.write(t, cell_output)
            return t + 1, out_ta, new_h, new_c

        time = tf.constant(0, dtype=tf.int32, name='time')
        loop_vars = (time, output_ta, init_h, init_c)

        result = tf.while_loop(lambda t, *_: t < config.num_steps, loop_func,
                               loop_vars)

        outputs = result[1].stack()
        outputs = tf.transpose(outputs, [1, 0, 2])

        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        logits = rnn_cell.linear(outputs, config.vocab_size, 'logits')
        logits = tf.reshape(logits, [tf.shape(self.xs)[0], config.num_steps,
                                     config.vocab_size])

        loss = tf.contrib.seq2seq.sequence_loss(logits, self.ys, tf.ones([
            tf.shape(self.xs)[0], config.num_steps], dtype=tf.float32))

        self.cost = loss

        optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
        #optimizer = tf.train.AdamOptimizer()
        if not config.clip:
            self.train_op = optimizer.minimize(loss)
        else:
            trainable_variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost,
                                              trainable_variables), 5)

            self.train_op = optimizer.apply_gradients(zip(grads,
                                                      trainable_variables))




