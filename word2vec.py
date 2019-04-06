#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from tempfile import gettempdir
import codecs
import re
import json

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

# 代码参考 https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

# step 1: 读取数据
filename = "data\\QuanSongCi.txt"


# Read the data into a list of strings.
def read_data(_filename):
    """读取文件中的汉字，建立词汇表"""
    with codecs.open(filename, 'rb', encoding='UTF-8') as f:
        _data = f.read()
        filtrate = re.compile(u'[^\u4E00-\u9FA5]')  # 非中文
        _data = filtrate.sub(r'', _data)
    return _data


vocabulary = read_data(filename)
print("vocabulary \n", vocabulary[:100])
print('Data size', len(vocabulary))

# Step 2: Build the dictionary and replace rare words with UNK token.
# 全宋词全文共6010个不同的单字符，这里只取出现次数最多的前5000个单字符
vocabulary_size = 5000


def build_dataset(words, n_words):
    """Process raw inputs into a dataset. 创建数据集"""
    # UNK表示出现次数排名于n_words之后的所有字
    count = [['UNK', -1]]
    # 计算words中，出现次数排名于n_words之前，各个字出现的次数
    kk = collections.Counter(words).most_common(n_words - 1)
    print(kk)
    count.extend(kk)
    _dictionary = dict()
    for word, _ in count:
        # dictionary key为字 value为
        _dictionary[word] = len(_dictionary)

    _data = list()
    _index = 0
    unk_count = 0
    for word in words:
        _index = _dictionary.get(word, 0)
        if _index == 0:  # dictionary['UNK']
            # 计算UNK数量
            unk_count += 1
        # 按顺序保存index
        _data.append(_index)
    count[0][1] = unk_count
    # 将dictionary的key和value调换
    reversed_dictionary = dict(zip(_dictionary.values(), _dictionary.keys()))
    return _data, count, _dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vocabulary_size-1).
#   This is the original text but words are replaced by their codes
# count - map of words(strings) to count of occurrences
# dictionary - map of words(strings) to their codes(integers)
# reverse_dictionary - maps codes(integers) to words(strings)
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

with open('data//dictionary.json', 'w') as f:
    json.dump(dictionary, f)
with open('data//reversed_dictionary.json', 'w') as f:
    json.dump(reverse_dictionary, f)

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
# 为Skip-gram模型生成training batch
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # 候选区域
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        # w != skip_window 即，非中心词，作为上下文
        context_words = [w for w in range(span) if w != skip_window]
        # random.sample()从序列context_words随机取num_skips个元素作为一个片段返回
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):  # 枚举
            batch[i * num_skips + j] = buffer[skip_window]  # batch是skip_window，即中心位置
            labels[i * num_skips + j, 0] = buffer[context_word]  # labels是上下文context_word
        if data_index == len(data):
            # buffer[:] = data[:span]
            for word in data[:span]:
                buffer.append(word)
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.
# 创建和训练Skip-gram模型

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector. embedding向量的维数
skip_window = 1  # How many words to consider left and right.窗口大小，即中心词左右上下文的长度
num_skips = 2  # How many times to reuse an input to generate a label.重复使用同一个输入生成label的次数
num_sampled = 64  # Number of negative examples to sample. 负样本的数量

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

graph = tf.Graph()

with graph.as_default():
    # Input data.
    # 表示源上下文字词的整数批次
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    # 表示目标字词的整数批次
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        # 嵌入矩阵，随机初始化
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 查询批次中每个源字词的向量
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        # 词汇表中每个字的权重和偏置
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    # 使用噪声对比训练目标来预测目标字词
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    # 计算梯度并更新，使用随机梯度下降法
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
# 训练模型
num_steps = 150001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    # 保存最终生成的embeding
    np.save('data//embedding.npy', final_embeddings)


# Step 6: Visualize the embeddings.
# 可视化学到的字词嵌入
# 使用 t-SNE 降维技术将字词嵌入投射到二维空间
# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    print('============== 开始绘图 =============== ')
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    print('============== 保存图片 =============== ')
    plt.savefig(filename)
    plt.show()


try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, 'data\\word2vector.png')

except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)
