# 宋词生成器

这是之前作业内容，见 [csdn-w11](https://github.com/ice-melt-CSDN/csdn-w11)
现打算重新梳理一下


## 1、`Word Embedding` 理解

Word Embedding 词嵌入。

__目的：__

将词语的`one-hot`高维稀疏表达，变换为维数相对低（仍是高维）的稠密表达。

因为将词语用`one-hot`表达，计算量大（维数非常高），并且词之间的关联无法体现。

__Word2Vector__：

`Word Embedding`的一种，Google 2013年开发的一个开源项目，是目前最成功有效简洁的词嵌入方式。

**Word2Vector**有2种机制：

`Skip-gram` 输入词去找和它关联的词，计算更快。

`CBOW` 输入关联的词去预测词。

 

# 2、`Word_Embedding.py`代码理解

本次作业进行`Word Embedding`使用的代码为`Word_Embedding.py`

`Word_Embedding`的代码主要分为6步：

#### (1)   读取数据

读取"QuanSongCi.txt"文件内容，并删掉所有非汉字的字符；

#### (2)   建立词汇表

统计输入数据中所有字出现的次数，并保留次数排名前5000的汉字，其余的汉字均视为“UNK”；

#### (3)   为`Skip-gram`模型生成`training batch`

生成`batch`为中心词，`labels`为上下文。

#### (4)   创建和训练Skip-gram模型

l.  随机初始化嵌入矩阵`embeddings`(`vocabulary_size*embedding_size`)，`vocabulary`中每个字都对应一个`embedding_size`维的字嵌入向量；

l.  通过`tf.nn.embedding_lookup`查询输入批次中每个字的嵌入向量；

l.  使用噪声对比训练目标来预测目标字词，计算`NCE loss`；

NCE loss：

1)      可以把多分类问题转化成二分类，大大提高计算速度；

2)      将所有单词分为两类，正例和负例，word2vec中只需给出上下文和相关的正例，tf.nn.nce_loss()中会自动生成负例。

(5)   训练模型

迭代步数取150001

运行计算图

通过np.save('embedding.npy', final_embeddings)保存最终生成的embeddings。

(6)   可视化学到的字词嵌入

使用 t-SNE 降维技术将字词嵌入投射到二维空间；

通过设置plt.rcParams['font.sans-serif'] = ['SimHei']，使matplotlib绘制的图能够正常显示中文。

 

# 3、字词嵌入图

![img](https://raw.githubusercontent.com/SophiaYuSophiaYu/Image/master/NoteImages/2019031401.png)

如图所示，可以看出图片中意义接近的词，如数字等（图中左下角），距离比较近（一这个数字是个特例，离其他数字比较远）。

 

# 4、RNN理解

RNN即Recurrent Neural Network，是循环神经网络，具有短期记忆能力，适用于文本和视频相关应用。

RNN网络模型结构如下图所示

![2019031402.jpg](https://github.com/SophiaYuSophiaYu/Image/blob/master/NoteImages/2019031402.jpg?raw=true)

​       RNN具有时许概念，每个数据产生时即会输入网络中，在网络内部存储之前输入的所有信息作为短期记忆。

​       本次作业使用的是RNN中的LSTM模型，如下图所示

![img](https://raw.githubusercontent.com/SophiaYuSophiaYu/Image/master/NoteImages/2019031403.png)

# 5、RNN代码

代码主要修改model.py、train.py和utils.py.

**model.py**

构建多层LSTM网络，用tf.nn.rnn_cell.MultiRNNCell创建self.rnn_layers层RNN，通过zero_state得到一个全0的初始状态，通过dynamic_rnn对cell在时间维度进行展开。

![img](https://raw.githubusercontent.com/SophiaYuSophiaYu/Image/master/NoteImages/2019031404.png)

**train.py**

读取训练数据，feed给模型，需要注意的是，每次训练完成后，把最后state的值再赋值回去供下次训练使用，代表了state在时间轴上的传递。

![img](https://raw.githubusercontent.com/SophiaYuSophiaYu/Image/master/NoteImages/2019031405.png)

 

**utils.py**

​       数据预处理，生成输入数据，data为文本中一段随机截取的文字，label为data对应的下一个标号的文字。以苏轼的江神子（江城子）为例：输入为 “老夫聊发少年”，则对应的label为"夫聊发少年狂"。

![img](https://raw.githubusercontent.com/SophiaYuSophiaYu/Image/master/NoteImages/2019031406.png)

![img](https://raw.githubusercontent.com/SophiaYuSophiaYu/Image/master/NoteImages/2019031407.png)

# 6、RNN模型训练心得

（1）一开始建立模型时，输入了之前做好的embedding矩阵，并不随模型一起训练，导致模型训练结果不好，几个epoch后，输出一直是UNK，令embedding矩阵随模型一起训练和更新就正常了；

 （2）batch_size设小一些，相对来说训练效果较好（2个epoch后就能输出看起来有一点意义的句子，而不是UNK或重复一个字），但是训练速度慢，一个epoch的step较多；

​       下图为batch_size为3时，训练2个epoch后的输出结果，可以看出，已经可以输出一些有意义的句子。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314172021319.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

如下图所示，batch_size为64时，2个epoch后的结果，可以看出只学习到了换行。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019031417213364.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

如下图所示，batch_size为64时，2个epoch后的结果，可以看出已经开始可以输出有意义的句子了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019031417220735.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

（3）模型运行结果

| num_epoch | batch_size | num_steps | total_loss | var_loss | time_cost |
| --------- | ---------- | --------- | ---------- | -------- | --------- |
| 30        | 3          | 64        | 4.79       | 0.03     | 13h       |
| 30        | 16         | 64        | 4.61       | 0.08     | 7h        |
| 30        | 64         | 64        | 5.10       | 0.37     | 4h        |

**batch_size 3**最终结果：（https://www.tinymind.com/executions/dgcf35wn）





![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314172233466.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314172256340.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314172322822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

 

 

**batch_size 16**最终结果：（https://www.tinymind.com/executions/cous9sgd）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314172351385.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314172417156.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314172438596.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

 

 

**batch_size 64**最终结果：（https://www.tinymind.com/executions/sshq3f1l）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314172500728.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314172518844.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019031417254139.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2t1YWl6aV9zb3BoaWE=,size_16,color_FFFFFF,t_70)

综上所述，可以看出，训练效果最好的模型参数为batch_size 16、num_steps 64，并且，从loss曲线可以看出，此模型若继续训练loss还会继续下降。

# 7、链接

模型运行链接：

batch_size为3：

<https://www.tinymind.com/executions/dgcf35wn>

batch_size为64：

<https://www.tinymind.com/executions/sshq3f1l>

batch_size为16：

<https://www.tinymind.com/executions/cous9sgd>

 

模型代码链接：

<https://gitee.com/SophiaYuSophiaYu/SongCiGenerator>

 

 
