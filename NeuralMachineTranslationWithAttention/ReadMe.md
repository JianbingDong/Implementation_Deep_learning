# 神经机器翻译 (Neural translation machine with attention)

参考 [Tensorflow Tutorial](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb)

数据集来自 [Anki](http://www.manythings.org/anki/)
*中文->English* 

Tensorflow教程中使用的是 eager_excution，此例中使用的是**先构造计算图，然后再调用Session的传统方法**。

-----
## 软件平台
+ Tensorflow 1.8.0
+ python 3.5.2
+ Windows 10

## 准备数据
1. Add a **start** and **end** token to each sentence
2. Clean the sentences by removing special characters.
3. Create a word index and reverse word index (dictionaries mapping from word->id and id->word)
4. Pad each sentence to a maximum length.

## 注意事项

word->idx以及idx->word比词汇表长度 **多一个**，因为后者没有统计 '<'pad>'
----
因此，在后续指定各全连接层输出神经元个数时，若需要使用词汇表的长度，则应该使用**len(word2idx)**或者**len(idx2word)**，不能使用 len(vocab)，这样的话。使用了len(vocab)就会导致 batch_y里面的值有可能会 等于 len(vocab)，则在计算损失函数的时候，***sparse_softmax_cross_entropy_with_logits*** 会输出 **NAN**。

## 使用方法
### 训练：
	调用 train() 主函数
### 测试：
	调用 translate() 主函数，其中的numSamples, embeddingDim, units参数需与训练时保持一致。




