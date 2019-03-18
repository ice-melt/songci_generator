# 宋词生成器

这是之前作业内容，见 [csdn-w11](https://github.com/ice-melt-CSDN/csdn-w11)

现打算重新梳理一下


## 1、`Word Embedding` 理解

`Word Embedding` 词嵌入。

### 目的：

将词语的`one-hot`高维稀疏表达，变换为维数相对低（仍是高维）的稠密表达。

因为将词语用`one-hot`表达，计算量大（维数非常高），并且词之间的关联无法体现。

### `Word2Vector`：

`Word Embedding`的一种，Google 2013年开发的一个开源项目，是目前最成功有效简洁的词嵌入方式。

`Word2Vector`有2种机制：

- `Skip-gram` 输入词去找和它关联的词，计算更快。

- `CBOW` 输入关联的词去预测词。

# 2、`Word_Embedding.py`代码理解

本次作业进行`Word Embedding`使用的代码为`Word_Embedding.py`

`Word_Embedding`的代码主要分为6步：

#### (1)   读取数据

读取`QuanSongCi.txt`文件内容，并删掉所有非汉字的字符；

#### (2)   建立词汇表

统计输入数据中所有字出现的次数，并保留次数排名前`5000`的汉字，其余的汉字均视为`UNK`；

#### (3)   为`Skip-gram`模型生成`training batch`

生成`batch`为中心词，`labels`为上下文。

#### (4)   创建和训练Skip-gram模型

-  随机初始化嵌入矩阵`embeddings`(`vocabulary_size*embedding_size`)，`vocabulary`中每个字都对应一个`embedding_size`维的字嵌入向量；
-  通过`tf.nn.embedding_lookup`查询输入批次中每个字的嵌入向量；
-  使用噪声对比训练目标来预测目标字词，计算`NCE loss`；
    
    `NCE loss`：

    - 可以把多分类问题转化成二分类，大大提高计算速度；
    - 将所有单词分为两类，正例和负例，`word2vec`中只需给出上下文和相关的正例，`tf.nn.nce_loss()`中会自动生成负例。

(5)   训练模型

迭代步数取150001

运行计算图

通过`np.save('embedding.npy', final_embeddings)`保存最终生成的`embeddings`。

(6)   可视化学到的字词嵌入

使用 `t-SNE` 降维技术将字词嵌入投射到二维空间；

通过设置`plt.rcParams['font.sans-serif'] = ['SimHei']`，使`matplotlib`绘制的图能够正常显示中文。

# 3、字词嵌入图

![PIC000][PIC000]

如图所示，可以看出图片中意义接近的词，如数字等（图中左下角），距离比较近（一这个数字是个特例，离其他数字比较远）。

 
# 4、RNN理解

`RNN`即`Recurrent Neural Network`，是循环神经网络，具有短期记忆能力，适用于文本和视频相关应用。

RNN网络模型结构如下图所示

![RNN网络模型结构](https://img-blog.csdnimg.cn/20190318164533612.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9pY2UtbWVsdC5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)
​       RNN具有时许概念，每个数据产生时即会输入网络中，在网络内部存储之前输入的所有信息作为短期记忆。

​       本次作业使用的是RNN中的LSTM模型，如下图所示
![LSTM模型](https://img-blog.csdnimg.cn/20190318164757754.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9pY2UtbWVsdC5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)
# 5、RNN代码

代码主要修改`model.py`、`train.py`和`utils.py`.

**model.py**

构建多层LSTM网络，用`tf.nn.rnn_cell.MultiRNNCell`创建`self.rnn_layers`层RNN，通过`zero_state`得到一个全`0`的初始状态，通过`dynamic_rnn`对`cell`在时间维度进行展开。
```python
# 堆叠多个lstm层，每层神经元个数为rnn_size个
def single_cell(lstm_size, keep_prob):
    cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return drop_cell

with tf.variable_scope('rnn'):
    # 用 tf.nn.rnn_cell.MultiRNNCell 创建 self.rnn_layers 层 RNN
    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell(self.rnn_size, self.keep_prob) for _ in range(self.rnn_layers)])
    self.initial_state = cell.zero_state(self.batch_size, tf.float32)
    # 通过dynamic_rnn对cell展开时间维度
    self.rnn_outputs, self.outputs_state_tensor = tf.nn.dynamic_rnn(cell, data, initial_state=self.initial_state)
    # self.rnn_outputs的维度是（batch，num_steps，rnn_size），concat后在time_step上减少一个维度
    seq_output = tf.concat(self.rnn_outputs, 1)   # 这一步不是必须的，删掉也OK
# flatten it
seq_output_final = tf.reshape(seq_output, [-1, self.rnn_size])   # reshape展开之后，维度变为（batch * num_steps）行，rnn_size列


with tf.variable_scope('softmax'):
    softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_words], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(self.num_words))
    logits = tf.matmul(seq_output_final, softmax_w) + softmax_b    # 最终shape是 （batch * num_steps）行，num_words列

tf.summary.histogram('logits', logits)
self.predictions = tf.nn.softmax(logits, name='predictions')
```
**train.py**

读取训练数据，feed给模型，需要注意的是，每次训练完成后，把最后state的值再赋值回去供下次训练使用，代表了state在时间轴上的传递。
```python
for epoc in range(1):
    # logging.debug('epoch [{0}]....'.format(epoc))
    state = sess.run(model.state_tensor)   # RNN的起始状态
    for x, y in utils.get_train_data(vocabulary_int, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):
        step += 1
        feed_dict = {model.X: x, model.Y: y, model.keep_prob: 0.85, model.state_tensor: state}

        gs, _, state, l, summary_string = sess.run(
            [model.global_step, model.optimizer, model.outputs_state_tensor,
             model.loss, model.merged_summary_op], feed_dict=feed_dict)
        summary_string_writer.add_summary(summary_string, gs)

        if gs % (max_steps // 10) == 0:
            logging.debug('step [{0}] loss [{1}]'.format(gs, l))

        if gs % (max_steps // 4) == 0:
            save_path = saver.save(sess, os.path.join(FLAGS.output_dir, "model.ckpt"), global_step=gs)

        if step >= max_steps:
            break
```

 

**utils.py**

​       数据预处理，生成输入数据，data为文本中一段随机截取的文字，label为data对应的下一个标号的文字。以苏轼的江神子（江城子）为例：输入为 “老夫聊发少年”，则对应的label为"夫聊发少年狂"。
```python
def get_train_data(vocabulary_int, batch_size, num_steps):
    """
    获取数据
    :param vocabulary_int:
    :param batch_size:batch 大小
    :param num_steps:num step 数量，即每个seq长度
    :return:
    """
    #  每次从data中取出（batch_size行，num_steps列）个元素，每个元素都是整数
    t = (len(vocabulary_int) // (batch_size*num_steps)) * (batch_size*num_steps)   # 防止data长度不能被batch_size、num_steps整除
    data_x = vocabulary_int[0:t]                 # 取出对应x长度数据
    data_y = np.zeros_like(data_x)
    data_y[:-1], data_y[-1] = data_x[1:], data_x[0]   # Y数据是x数据右移一位的结果

    x_batch = data_x.reshape((batch_size, -1))      # reshape成batch_size行，每一行再拆分成若干个num_steps
    y_batch = data_y.reshape((batch_size, -1))      # reshape成batch_size行，每一行再拆分成若干个num_steps

    while True:
        for n in range(0, x_batch.shape[1], num_steps):  # 以num_steps步进
            x = x_batch[:, n:n + num_steps]           # 行表示batch_size，列表示num_steps
            y = y_batch[:, n:n + num_steps]           # 行表示batch_size，列表示num_steps
            yield x, y                      # 采用生成器

```

# 6、RNN模型训练心得
RNN的网络单元和其他的神经网络基本相同都是由隐层，输入和输出层组成，但是RNN有在时间维度堆叠，使其可以将数据前后的相关性在网络结构中学习出来，即具有记忆功能

word embeding是将一堆词组成的词汇表中的单词或短语映射成实数构成的向量上。这样的好处是通过简单的余弦函数，就可以计算两个单词之间的相关性

RNN训练过程中每个batch会喂多个time_step的数据,最基础的RNN网络由于不断的对同一个参数矩阵进行梯度更新，很容易产生梯度消失或梯度爆炸的情况，对于梯度降低过大可以采用直接人为限制梯度的大小的方式解决(可以使loss降低)，除此之外调整网络结构是一个更好的方法，LSTM网络可以将BP中的乘法转化成加法，从而大大减小梯度的衰减速度


- 在tinymind运行log的输出截图

![PIC002][PIC002]

> 输出的结果有点诗的感觉 但是和真正的诗词又有很大区别，输出明显能看出网络能够得到上下文相关的信息，但是还需要进一步提升性能
#### 7、链接

gitee: [https://gitee.com/ice-melt/eleventh_weeks_homework](https://gitee.com/ice-melt/eleventh_weeks_homework)
github: [https://github.com/ice-melt-CSDN/csdn-w11]( [csdn-w11](https://github.com/ice-melt-CSDN/csdn-w11))
tinymind: [https://www.tinymind.com/executions/8a50l7dz](https://www.tinymind.com/executions/8a50l7dz)
  
------
 
[PIC000]: https://raw.github.com/ice-melt/picture-set/master/csdn_week11_homework-pic/tsne.png
[PIC001]: https://raw.github.com/ice-melt/picture-set/master/csdn_week11_homework-pic/dataset.png
[PIC002]: https://raw.github.com/ice-melt/picture-set/master/csdn_week11_homework-pic/runLog.png
[PIC003]: https://raw.github.com/ice-melt/picture-set/master/csdn_week11_homework-pic/param.png

