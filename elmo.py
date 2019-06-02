import tensorflow as tf
from tensorflow.contrib import rnn


class ELMo(object):
    def __init__(self, batch_size, hidden_size, vocab_size):
        """
        ELMo is a deep contextualized word representation that models both
         (1) complex characteristics of word use (e.g., syntax and semantics),
         (2) how these uses vary across linguistic contexts (i.e., to model polysemy).
        :param batch_size: the size of each batch
        :param hidden_size: the size of LSTM's hidden layer 
        :param vocab_size: the size of all vocabulary 
        """
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.layer_num = 2

    def elmo(self, x, length, dropout=1.):
        """
        the network of ELMo
        :param x: the sequence of each word's id, shape: (batch_size, max_length), dtype: int32
        :param length: the valid length of each sentence, shape: (batch_size, max_length) , dtype: int32
        :param dropout:
        :return: the representation of each words
        """
        with tf.variable_scope('elmo_scope', reuse=tf.AUTO_REUSE):
            def lstm_cell(hidden_size, cell_id=0):
                # generator of lstm cell
                cell = rnn.LSTMCell(hidden_size, reuse=tf.AUTO_REUSE, name="lstm_cell%d" % cell_id)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout)
                return cell

            # word embedding
            embeddings = tf.Variable(tf.random_normal(
                [self.vocab_size, self.hidden_size],
                mean=0.0, stddev=0.1, dtype=tf.float32), name="embedding")
            # self.weights of outputs from every layers
            self.weights = tf.Variable(
                tf.random_normal([2 * self.layer_num + 1], mean=0.0, stddev=0.1, dtype=tf.float32),
                name="weights")

            # forward and backward vectors that are inputted the lstm network
            self.X_fw = tf.nn.embedding_lookup(embeddings, x)
            X_bw = tf.reverse_sequence(self.X_fw,
                                       seq_lengths=length,
                                       seq_axis=1, batch_axis=0)

            # forward layer 1
            fw_1 = lstm_cell(self.hidden_size, 0)
            fw_zero_1 = fw_1.zero_state(self.batch_size, tf.float32)
            self.fw_output_1, fw_state_1 = tf.nn.dynamic_rnn(fw_1, self.X_fw,
                                                        sequence_length=length,
                                                        initial_state=fw_zero_1,
                                                        dtype=tf.float32)
            # forward layer 2
            fw_2 = lstm_cell(self.hidden_size, 1)
            fw_zero_2 = fw_2.zero_state(self.batch_size, tf.float32)
            self.fw_output_2, fw_state_2 = tf.nn.dynamic_rnn(fw_2, self.fw_output_1,
                                                        sequence_length=length,
                                                        initial_state=fw_zero_2,
                                                        dtype=tf.float32)
            # backward layer 1
            bw_1 = lstm_cell(self.hidden_size, 2)
            bw_zero_1 = bw_1.zero_state(self.batch_size, tf.float32)
            _bw_output_1, bw_state_1 = tf.nn.dynamic_rnn(bw_1, X_bw,
                                                         sequence_length=length,
                                                         initial_state=bw_zero_1,
                                                         dtype=tf.float32)

            # backward layer 2
            bw_2 = lstm_cell(self.hidden_size, 3)
            bw_zero_2 = bw_2.zero_state(self.batch_size, tf.float32)
            _bw_output_2, bw_state_2 = tf.nn.dynamic_rnn(bw_2, _bw_output_1,
                                                         sequence_length=length,
                                                         initial_state=bw_zero_2,
                                                         dtype=tf.float32)
            # reverse the output of backward layer
            self.bw_output_1 = tf.reverse_sequence(_bw_output_1,
                                              seq_lengths=length,
                                              seq_axis=1, batch_axis=0)
            self.bw_output_2 = tf.reverse_sequence(_bw_output_2,
                                              seq_lengths=length,
                                              seq_axis=1, batch_axis=0)
            # calculate the softmax of self.weights
            _weights = tf.nn.softmax(self.weights)

            # get the representation of each words
            output = _weights[0] * self.X_fw + \
                     _weights[1] * self.fw_output_1 + \
                     _weights[2] * self.fw_output_2 + \
                     _weights[3] * self.bw_output_1 + \
                     _weights[4] * self.bw_output_2

            return output
