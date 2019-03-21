#!/usr/bin/env python
# visit http://tool.lu/pyc/ for more information
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import tensorflow as tf
import thumt.interface as interface
from tensorflow.contrib.seq2seq.python.ops import *
import thumt.cyn.rnn_cell as rnn_cell
import thumt.cyn.attention as attention
import thumt.layers as layers


def dynamic_lstm(cell, inputs, initial_state=None,
            scope="dynamic_lstm"):
    with tf.variable_scope(scope):
        batch_size = tf.shape(inputs)[0]
        time_step = tf.shape(inputs)[1]
        output_size = cell.output_size

        if initial_state is None:
            initial_state = cell.zero_state(batch_size, tf.float32)

        input_ta = tf.TensorArray(tf.float32, time_step,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(tf.float32, time_step,
                                   tensor_array_name="output_array")
        input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

        def func(t, sta, out):
            new_output, new_state = cell(input_ta.read(t), sta)
            out = out.write(t, new_output)
            return t + 1, new_state, out

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, initial_state, output_ta)
        loop_outputs = tf.while_loop(lambda t, *_: t < time_step, func,
                                     loop_vars)

        outputs = loop_outputs[2].stack()
        outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, loop_outputs[1]


def decoder(cell, inputs, atten, initial_state=None,
            scope="decoder"):
    with tf.variable_scope(scope):
        batch_size = tf.shape(inputs)[0]
        time_step = tf.shape(inputs)[1]
        output_size = cell.output_size

        if initial_state is None:
            initial_state = cell.zero_state(batch_size, tf.float32)

        input_ta = tf.TensorArray(tf.float32, time_step,
                                  tensor_array_name="input_array")
        output_ta = tf.TensorArray(tf.float32, time_step,
                                   tensor_array_name="output_array")
        value_ta = tf.TensorArray(tf.float32, time_step,
                                  tensor_array_name="value_array")
        input_ta = input_ta.unstack(tf.transpose(inputs, [1, 0, 2]))

        def func(t, sta, out, val):
            _, context = atten(sta[0][1])
            (new_output, new_state) = cell(input_ta.read(t), sta, context)
            out = out.write(t, new_output)
            val.write(t, context)
            return t + 1, new_state, out, val

        time = tf.constant(0, dtype=tf.int32, name="time")
        loop_vars = (time, initial_state, output_ta, value_ta)
        loop_outputs = tf.while_loop(lambda t, *_: t < time_step, func,
                                     loop_vars)

        outputs = loop_outputs[2].stack()
        outputs = tf.transpose(outputs, [1, 0, 2])
        values = loop_outputs[3].stack()
        values = tf.transpose(values, [1, 0, 2])

    return outputs, loop_outputs[1], values


def model_graph(features, mode, params):
    svocab_size = len(params.vocabulary["source"])
    tvocab_size = len(params.vocabulary["target"])
    num_layers = 3

    with tf.variable_scope("source_embedding"):
        embedding = tf.get_variable("embedding", [
            svocab_size,
            params.embedding_size])
        src_inputs = tf.nn.embedding_lookup(embedding, features["source"])
    with tf.variable_scope("target_embedding"):
        embedding = tf.get_variable("embedding", [
            svocab_size,
            params.embedding_size])
        tgt_inputs = tf.nn.embedding_lookup(embedding, features["target"])
        tgt_inputs = tf.pad(tgt_inputs, [[0, 0], [1, 0], [0, 0]])
        tgt_inputs = tgt_inputs[:, :-1, :]

    if mode == "train":
        src_inputs = tf.nn.dropout(src_inputs, params.keep_prob)
        tgt_inputs = tf.nn.dropout(src_inputs, params.keep_prob)

    # encoder
    with tf.variable_scope("encoding"):
        with tf.variable_scope("bidirection"):
            cell_fw = rnn_cell.LSTMCell(params.hidden_size)
            cell_bw = rnn_cell.LSTMCell(params.hidden_size)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob
                                                    =params.keep_prob)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob
                                                    =params.keep_prob)

            (output_fw, _) = dynamic_lstm(cell_fw, src_inputs, scope="forward")

            rever_inputs = tf.reverse_sequence(src_inputs, features[
                "source_length"], batch_axis=0, seq_axis=1)
            (output_bw, _) = dynamic_lstm(cell_bw, rever_inputs,
                                          scope="backward")
            output_bw = tf.reverse_sequence(output_bw, features["source_length"]
                                            , batch_axis=0, seq_axis=1)

        with tf.variable_scope("unidirection"):
            en_inputs = tf.concat([output_fw, output_bw], axis=-1)
            i = num_layers - 1
            cells = []
            while i:
                cell = rnn_cell.LSTMCell(params.hidden_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob
                                                     =params.keep_prob)
                cell = tf.contrib.rnn.ResidualWrapper(cell)
                cells.append(cell)
                i -= 1
            cells = rnn_cell.MultiRNNCell(cells)
            en_outputs, en_state = dynamic_lstm(cells, en_inputs, scope="uni")

    # decoder
    with tf.variable_scope("decoding"):
        i = num_layers
        atten = attention.Attention(params.hidden_size, en_outputs)
        cells = []
        while i:
            cell = rnn_cell.LSTMCell(params.hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob
                                                 =params.keep_prob)
            cells.append(cell)
            i -= 1
        cells = rnn_cell.MultiRNNCell(cells)
        de_outputs, de_state, values = decoder(cells, tgt_inputs, atten,
                                               scope="decoding")

    if mode == "infer":
        logits = rnn_cell.linear(de_outputs[:, -1,:], tvocab_size, "logit")
        logits = tf.nn.log_softmax(logits)
        return logits
    else:
        de_outputs = tf.reshape(de_outputs, [-1, params.hidden_size])
        logits = rnn_cell.linear(de_outputs, tvocab_size, "logit")

    labels = tf.reshape(features["target"], [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=labels)
    loss = tf.reshape(loss, tf.shape(features["target"]))
    mask = tf.to_float(tf.sequence_mask(features["target_length"],
                                        maxlen=tf.shape(features["target"])[1]))
    if mode == "eval":
        return -tf.reduce_sum(loss * mask, axis=1)
    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return loss


class Gnmt(interface.NMTModel):

    def __init__(self, params, scope="gnmt"):
        super(Gnmt, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = self.parameters
            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def inference_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            params.dropout = 0.0
            params.use_variational_dropout = False
            params.label_smoothing = 0.0

            with tf.variable_scope(self._scope):
                log_prob = model_graph(features, "infer", params)

            return log_prob

        return inference_fn

    @staticmethod
    def get_name():
        return "gnmt"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            # vocabulary
            pad="<pad>",
            unk="<unk>",
            eos="<eos>",
            bos="<eos>",
            append_eos=False,
            # model
            rnn_cell="=LstmCell",
            embedding_size=620,
            hidden_size=1000,
            maxnum=2,
            # regularization
            dropout=0.2,
            use_variational_dropout=False,
            label_smoothing=0.1,
            constant_batch_size=True,
            batch_size=128,
            max_length=60,
            clip_grad_norm=5.0,
            keep_prob=1.0,
            num_layers = 8
        )

        return params

