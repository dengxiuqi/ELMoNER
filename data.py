from collections import Counter
import numpy as np


TRAIN_DATA = "data/example.train"
TEST_DATA = "data/example.test"
DEV_DATA = "data/example.dev"


class NERData(object):
    def __init__(self, batch_size, max_length=128):
        """
        :param batch_size: 每个batch的大小
        :param max_length: 每条语料的最大长度
        """
        self.batch_size = batch_size
        self.max_length = max_length

        self.train_data = self._load(TRAIN_DATA)
        self.test_data = self._load(TEST_DATA)
        self.dev_data = self._load(DEV_DATA)
        self.word_map, self.entity_map = self._bulid_map(self.train_data)
        self.vocab_size = len(self.word_map) + 1

        self.train_X, self.train_length, self.train_targets, self.train_weights = self._data_encode(self.train_data)
        self.test_X, self.test_length, self.test_targets, self.test_weights = self._data_encode(self.test_data)
        self.dev_X, self.dev_length, self.dev_targets, self.dev_weights = self._data_encode(self.dev_data)

        self.epoch = 0
        self.train_cursor = 0
        self.test_cursor = 0

    def get_train_batch(self):
        """从训练集中获取一个batch"""
        if self.train_cursor + self.batch_size > len(self.train_X):
            self.epoch += 1
            self.train_cursor = 0
        X = self.train_X[self.train_cursor: self.train_cursor + self.batch_size]
        length = self.train_length[self.train_cursor: self.train_cursor + self.batch_size]
        targets = self.train_targets[self.train_cursor: self.train_cursor + self.batch_size]
        weights = self.train_weights[self.train_cursor: self.train_cursor + self.batch_size]
        self.train_cursor += self.batch_size
        return X, length, targets, weights

    def get_test_batch(self):
        """从测试集中获取一个batch"""
        if self.test_cursor + self.batch_size > len(self.test_X):
            self.test_cursor = 0
        X = self.test_X[self.test_cursor: self.test_cursor + self.batch_size]
        length = self.test_length[self.test_cursor: self.test_cursor + self.batch_size]
        targets = self.test_targets[self.test_cursor: self.test_cursor + self.batch_size]
        weights = self.test_weights[self.test_cursor: self.test_cursor + self.batch_size]
        self.test_cursor += self.batch_size
        return X, length, targets, weights

    def get_dev_data(self, batch_size):
        """从验证集中获取一个batch"""
        X = self.dev_X[: batch_size]
        length = self.dev_length[:batch_size]
        targets = self.dev_targets[:batch_size]
        return X, length, targets

    def word2id(self, word):
        """将字词转化为id"""
        return self.word_map.get(word, 0)

    def entity2id(self, entity):
        """将命名实体转化为id"""
        return self.entity_map.get(entity, 0)

    def id2entity(self, Id):
        """将id还原为实体名"""
        return self.words[Id]

    def sentence_encode(self, sentence):
        """对一个句子编码，将其转化为可以直接输入elmo网络的数据"""
        word_ids = [0] * self.max_length
        for k, w in enumerate(sentence):
            word_ids[k] = self.word2id(w)
        X = np.reshape(np.array(word_ids, dtype=np.int32), (1, -1))
        length = np.array([len(sentence)], dtype=np.int32)
        return X, length

    def entities_decode(self, entities):
        """对一段实体序列进行解码，用于将网络输出的结果还原维实体名称"""
        result = []
        for e in entities:
            result.append(self.entities[e])
        return result

    def _load(self, path):
        """加载语料库"""
        with open(path, "r", encoding="utf8") as f:
            raw_data = f.read()     # 原始数据
        data = []
        raw_data = raw_data.split("\n\n")   # 将两条不同的语料分开
        for k, d in enumerate(raw_data):
            contents = []
            for token in d.split("\n"):     # 文字与实体标签
                try:
                    w, e = token.split()
                    contents.append((w, e))
                except:
                    pass
            if 0 < len(contents) < self.max_length:     # 只保留长度在0-max_length之间的语料
                data.append(contents)
        print("成功加载语料%s, 语料数量%d" % (path, len(data)))
        return data

    def _bulid_map(self, data):
        """构建词语/实体索引字典"""
        words, entities = [], []
        for d in data:
            for w, e in d:
                words.append(w)
                entities.append(e)
        words_total = Counter(words).most_common()      # 按出现的频率从高到低排列
        entities_total = Counter(entities).most_common()
        word_map = dict([(w[0], k+1) for k, w in enumerate(words_total)])
        entity_map = dict([(e[0], k+1) for k, e in enumerate(entities_total)])
        self.words = ["UKN"] + [w for w, _ in words_total]
        self.entities = ["UKN"] + [e for e, _ in entities_total]
        return word_map, entity_map

    def _data_encode(self, data):
        """对原始的训练集/测试集/验证集编码成向量形式"""
        X, length, targets = [], [], []
        weights = np.zeros(shape=(len(data), self.max_length), dtype=np.float32)
        for i, d in enumerate(data):
            word_ids, entity_ids = [0] * self.max_length, [0] * self.max_length
            for j in range(len(d)):
                word_ids[j] = self.word2id(d[j][0])
                entity_ids[j] = self.entity2id(d[j][1])
            length.append(len(d))
            weights[i, :len(d)] = 1.
            X.append(word_ids)
            targets.append(entity_ids)
        X = np.array(X, dtype=np.int32)
        length = np.array(length, dtype=np.int32)
        targets = np.array(targets, dtype=np.int32)
        return X, length, targets, weights


