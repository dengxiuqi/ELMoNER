import tensorflow as tf
from tensorflow.contrib import seq2seq
from elmo import ELMo
from data import NERData
import os

total_epoch = 5000
hidden_size = 200
vocab_size = 5000
max_length = 128
entity_class = 7

lr = 1e-4
batch_size = 256

ner = NERData(batch_size, max_length)
elmo = ELMo(batch_size, hidden_size, vocab_size)


def network(X):
    w = tf.get_variable("fcn_w", [1, hidden_size, entity_class + 1])
    b = tf.get_variable("fcn_b", [entity_class + 1])
    # 这里输出维度用entity_class + 1而不是entity_class，因为输出里除了7类实体，还有一类用来表示每个句子补齐的<PAD>位
    w_tile = tf.tile(w, [batch_size, 1, 1])

    logists = tf.nn.softmax(tf.nn.xw_plus_b(X, w_tile, b), name="logists")
    return logists


def train():
    X = tf.placeholder(shape=[batch_size, max_length], dtype=tf.int32, name="X")
    length = tf.placeholder(shape=[batch_size], dtype=tf.int32, name="length")
    targets = tf.placeholder(shape=[batch_size, max_length], dtype=tf.int32, name="targets")
    weights = tf.placeholder(shape=[batch_size, max_length], dtype=tf.float32, name="weights")
    dropout = tf.placeholder(shape=[], dtype=tf.float32, name="dropout")

    elmo_vector = elmo.elmo(X, length, dropout)
    logists = network(elmo_vector)

    seq_loss = seq2seq.sequence_loss(logists, targets, weights)
    # optimizer = tf.train.AdamOptimizer(lr).minimize(seq_loss)

    trainableVariables = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(lr)
    grads, a = tf.clip_by_global_norm(tf.gradients(seq_loss, trainableVariables), 5)
    train_op = optimizer.apply_gradients(zip(grads, trainableVariables))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model_dir = "./model"
        saver = tf.train.Saver(max_to_keep=10)
        if not os.path.exists(model_dir):   # 检查./model路径是否存在
            os.mkdir(model_dir)             # 不存在就创建路径
            print("create the directory: %s" % model_dir)
        check_point = tf.train.get_checkpoint_state(model_dir)
        if check_point and check_point.model_checkpoint_path:
            saver.restore(sess, check_point.model_checkpoint_path)
            print("restored %s" % check_point.model_checkpoint_path)
        else:
            print("no checkpoint found!")

        step = 0
        total_loss = 0
        while ner.epoch < total_epoch:
            _X, _length, _targets, _weights = ner.get_train_batch()
            fd = {X: _X, length: _length, targets: _targets, weights: _weights, dropout: .75}
            _, l = sess.run([train_op, seq_loss], feed_dict=fd)
            total_loss += l
            if step % 100 == 0:
                print("epoch:", ner.epoch, "step:", step, "loss:", total_loss / 100)
                total_loss = 0

            if step % 1000 == 0:
                _X, _length, _targets, _weights = ner.get_test_batch()
                fd = {X: _X, length: _length, targets: _targets, weights: _weights, dropout: 1.}
                l, result = sess.run([seq_loss, logists], feed_dict=fd)
                result = result.argmax(axis=2)
                ner_num = 0
                for i in range(batch_size):
                    for j in range(_length[i]):
                        if result[i, j] != 1:
                            ner_num += 1
                print("平均每句有标注的实体数:", ner_num / batch_size)
                print("test_loss:", l)

            if step % 10000 == 0:
                saver.save(sess, model_dir + '/model', global_step=step)  # 保存模型
                print("saving...")

            step += 1


if __name__ == "__main__":
    train()
