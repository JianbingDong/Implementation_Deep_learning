# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:08:16 2019

@author: Jianbing_Dong
"""

#%%
import tensorflow as tf

from matplotlib import pyplot as plt

import re
import numpy as np
import os
import time

#%%
#预处理
def preprocess_sentence(sentence):
    """
    对句子进行预处理，在每个词以及标点之间添加空格，并去除一些少见的标点符号，
    然后在句子前后加上 开始 和 结束 的标志。
    """
    sentence = sentence.strip()
    #英文
    if ord(sentence.strip()[0]) in range(97, 122 + 1) or\
        ord(sentence.strip()[0]) in range(65, 90 + 1):
            
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r"[^a-zA-Z?.!,¿0-9]+", " ", sentence)
        
        sentence = "<start> " + sentence + "<end>"
    else:
        #中文
        sentence = re.sub(r'([·《》“”])', "", sentence)
        sentence = [word for word in sentence]
        sentence = " ".join(sentence)
    
        sentence = "<start> " + sentence + " <end>"
    
    return sentence
    
    
def create_dataset(fileName, numSamples):
    """
    从文本当前读取内容，并对句子进行处理，返回 [english, 中文]的形式.
    #arguments:
        fileName: string, where to find the training file.
        numSamples: how much samples used to train.
    #returns:
        word_pairs: [[english, madrian],
                     [english, madrian]]
    """
    #不使用此种编码的话，读取文件之后，在开始位置会有\ufeff
    with open(fileName, 'r', encoding='utf-8-sig') as txtfile: 
        lines = txtfile.read().strip().split('\n')
        
        word_pairs = [[preprocess_sentence(sentence) for sentence in line.split('\t')]
                       for line in lines[:numSamples]]

    return word_pairs
        

class LanguageIndex(object):
    """
    构建一个数据结构来保存词汇表，word->index以及index->word的映射表
    """
    def __init__(self, language):
        self.lang = language #某种语言的所有句子
        self.word2idx = {} #word -> index
        self.idx2word = {} #index -> word
        self.vocab = set() #vocabulary词汇表
        
        self.create_index()
        
    def create_index(self):
        #建立词汇表
        for phrase in self.lang: 
            self.vocab.update(phrase.split(' '))
        
        #排序
        self.vocab = sorted(self.vocab)
        
        #word->index
        self.word2idx['<pad>'] = 0 #空格 对应的索引值
        for index, word in enumerate(self.vocab): #返回索引和内容
            self.word2idx[word] = index + 1

        #index->word
        for word, index in self.word2idx.items():
            self.idx2word[index] = word


#%%
def maxLength(tensors):
    """
    用于找出所有向量中的最长向量的长度
    """
    return max(len(tensor) for tensor in tensors)
    
def load_dataset(fileName, numSamples):
    """
    用于生成训练数据
    #arguments:
        fileName: string, 
        numSamples: integer.
    #returns:
        input_tensor, target_tensor: np.ndarray with shape [numSamples, maxLength]
        engLang, manLang: the data structure for each language.
        maxL_input, maxL_target: integer, the max length for each language.
    """
    wordPairs = create_dataset(fileName, numSamples)
    english, mandarin = list(zip(*wordPairs)) #将English与普通话分别取出

    #建立词汇表及映射表
    engLang = LanguageIndex(english)
    manLang = LanguageIndex(mandarin)
    
    #对语句进行向量化，即将字符转为对应的数字
    #英文作为输入 [numSamples, tensor]
    input_tensor = [[engLang.word2idx[word] for word in en.split(' ')] 
                     for en in english]
                     
    #中文作为输出 [numSamples, tensor]
    target_tensor = [[manLang.word2idx[word] for word in man.split(' ')]
                      for man in mandarin]
         
    #分别获取两种语言的最长句子长度
    maxL_input = maxLength(input_tensor)
    maxL_target = maxLength(target_tensor)
    
    #将序列补0到相同长度, np.ndarray with shape [numSamles, maxLength]
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=maxL_input,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=maxL_target,
                                                                  padding='post')
    
    return input_tensor, target_tensor, engLang, manLang, maxL_input, maxL_target
    
    
def train_test_split(input_, target, test_size=0.2):
    """
    用于拆分数据为训练集和验证集
    #arguments:
        input_: 输入数据，np.ndarray with shape [numsamples, maxlength]
        target: 标签数据, np.ndarray with shape [numsameples, maxlengthOfTarget]
        test_size: float, 用于指定多少数据做验证集
    #returns:
        input_train, input_val, target_train, target_val
    """
    numSamples = input_.shape[0]
    numTest = int(test_size * numSamples) + 1
    
    allIndex = range(numSamples)
    testIndex = sorted(np.random.choice(allIndex, size=numTest, replace=False))
    trainIndex = sorted(list(set(allIndex).difference(set(testIndex))))
    
    input_train = input_[trainIndex]
    input_val = input_[testIndex]

    target_train = target[trainIndex]
    target_val = target[testIndex]

    return input_train, input_val, target_train, target_val
    
#%%

class Model(object):
    
    def __init__(self, numSamples, batchSize, embeddingDim, units,
                 epochs, learningRate, engLanguage, manLanguage,
                 input_, target,
                 ifTraining=True):
        
        tf.reset_default_graph()
        
        self.numSamples = numSamples
        self.batchSize = batchSize
        self.embeddingDim = embeddingDim
        self.units = units

        self.epochs = epochs
        self.learningRate = learningRate
        self.ifTraining = ifTraining
        
        self.engLang = engLanguage
        self.manLang = manLanguage
        self.input_ = input_
        self.target = target
        
    def dataBatch(self, input_, target, shuffle=True):
        """
        此函数用于产生batch数据
        #arguments:
            input_, target: np.ndarray, 训练数据及对应标签 [numSamples, maxLength]
            epochs: integer,训练总轮数
            shuffle: bool, 是否对数据集进行打乱
        #returns:
            batch_x, batch_y: np.ndarray with shape[batchSize, maxlength]
        """
        allIndex = list(range(input_.shape[0]))
        
        for epoch in range(self.epochs):
            if shuffle:
                np.random.shuffle(allIndex)
                
            hidden = self.initialHiddenState(shape=(self.batchSize, self.units))
                
            for begin in range(0, input_.shape[0], self.batchSize):
                end = begin + self.batchSize
                if end >= input_.shape[0]:
                    #end = input_.shape[0]
                    continue
                    
                x_index = allIndex[begin : end]
                y_index = allIndex[begin : end]

                batch_x = input_[x_index]
                batch_y = target[y_index]

                yield batch_x, batch_y, hidden
                
                
    def inference(self, input_, target, hiddenState):
        """
        inference with the model to generate output.
        e is short for encoder
        d is short for decoder
        a is short for attention
        """
        if not self.ifTraining:
            returnPredictions = [] #用于记录每次预测的结果
            
        allpre = []
        loss = 0
        self.encoderEmbedding = tf.get_variable(name='encoder_embedding', 
                        shape=(len(self.engLang.word2idx), self.embeddingDim),
                            dtype=tf.float32) #(vocabSize, embeddingDim)
        self.decoderEmbedding = tf.get_variable(name='decoder_embedding',
                        shape=(len(self.manLang.word2idx), self.embeddingDim),
                            dtype=tf.float32) #(vocabSize, embeddingDime)
        
        with tf.variable_scope('encoder'):
            e_x = tf.nn.embedding_lookup(self.encoderEmbedding, input_) #[batchSize, maxLength, embeddingDim]
            e_gru = self.gRU(self.units)
            e_output, e_state = e_gru(e_x, initial_state = hiddenState)
            #[batchSize, maxLength, units], [batchSize, units]
            
        with tf.variable_scope('decoder_and_attention'):
            d_hidden = e_state
            
            if self.ifTraining:
                d_input = tf.expand_dims([self.manLang.word2idx['<start>']] *\
                                             self.batchSize, 1) #[batchSize, 1]
            else:
                d_input = tf.expand_dims([self.manLang.word2idx['<start>']], 0)
                
            #training
            for t in range(1, self.target.shape[1]):
                hidden_with_time_axis = tf.expand_dims(d_hidden, 1)
                #[batchSize, 1, units]
                
                w1 = tf.layers.dense(e_output, self.units)
                w2 = tf.layers.dense(hidden_with_time_axis, self.units)
                
                score = tf.layers.dense(tf.nn.tanh(w1 + w2), 1)
                #[batchSize, maxLength, 1]
                attention_weights = tf.nn.softmax(score, axis=1)
                #[batchSize, maxLength, 1]
                
                context_vector = tf.reduce_sum(attention_weights * e_output, axis=1)
                #[batchSize, units]

                d_x = tf.nn.embedding_lookup(self.decoderEmbedding, d_input) 
                #[batchSize, 1, embeddingDim]
            
                d_x = tf.concat([tf.expand_dims(context_vector, 1), d_x], axis=-1)
                #[batchSize, 1, units + embeddingDim]
                d_gru = self.gRU(self.units)
            
                d_output, d_state = d_gru(d_x)
                #[batchSize, 1, units], [batchSize, units]
            
                d_output = tf.reshape(d_output, (-1, d_output.shape[2]))
                #[batchSize, units]
            
                d_x = tf.layers.dense(d_output, len(self.manLang.word2idx))
                #[batchSize, vocabSize]
                
                #return
                d_hidden = d_state
                
                prediction = d_x
                allpre.append(prediction)
                if not self.ifTraining:
                    predicted_id = tf.argmax(prediction[0])
                    returnPredictions.append(predicted_id)
                        
                    d_input = tf.expand_dims([predicted_id], 0)
                		
                else:
                    loss += self.build_cost(target[:, t], prediction)
                    
                    d_input = tf.expand_dims(target[:, t], 1)
                    #[batchSize, 1]
                
        if self.ifTraining:
            return loss, allpre
        else:
            return returnPredictions
                
        
    def build_cost(self, real, pred):
        with tf.name_scope('loss'):
            #real = tf.one_hot(real, depth=pred.shape[-1])
            mask = 1 - np.equal(real, 0)
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
            #loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(labels=real, logits=pred)

        return tf.reduce_mean(loss_)
        
        #TODO:
    def build_optimizer(self, loss, variables):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grad_and_vars = optimizer.compute_gradients(loss, var_list=variables)
            
            clipped_grads_vars = [(tf.clip_by_value(grad, -1e-3, 1e-3), var)
                                  for grad, var in grad_and_vars]
            train_op = optimizer.apply_gradients(clipped_grads_vars)
            
        return train_op
        
        
    def train(self, save_n, show_n, savePath):
        
        trainGraph = tf.Graph()
        with trainGraph.as_default():
            x = tf.placeholder(dtype=tf.int32, name='input_',
                               shape=(self.batchSize, self.input_.shape[1]))
            y = tf.placeholder(dtype=tf.int32, name='target',
                               shape=(self.batchSize, self.target.shape[1]))
            h = tf.placeholder(dtype=tf.float32, name='hiddenstate',
                               shape=(self.batchSize, self.units))
            
            loss, allPre = self.inference(x, y, h)
            batch_loss = (loss / int(self.target.shape[1]))
            
            variables = tf.global_variables()
            
            train_op = self.build_optimizer(batch_loss, variables)
            
            saver = tf.train.Saver(var_list=variables, max_to_keep=5)
            
            init_op = tf.group(tf.local_variables_initializer(),
                            tf.global_variables_initializer())
            
            
        with tf.Session(graph = trainGraph) as sess:
            sess.run(init_op)
			
            sess.graph.finalize()
            
            step = 0
            for batch_x, batch_y, hidden in self.dataBatch(self.input_, 
                                        self.target, shuffle=True):
                _, l, preTrain = sess.run([train_op, batch_loss, allPre], 
                         feed_dict={x: batch_x, y: batch_y, h: hidden})
                #print(preTrain)
                
                if step % save_n == 0:
                    saver.save(sess, save_path=savePath + r'neuralTranslation', 
                               global_step=step, write_meta_graph=False)
                    
                if step % show_n == 0:
                    print("step: %i, loss: %.2f" %(step, l))
					
                step += 1
                
                
    def predict(self, sentence, modelPath):
        sentence = preprocess_sentence(sentence) #对输入进行预处理
        
        inputs = [self.engLang.word2idx[i] for i in sentence.split(' ')] #转换为数字向量
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], 
                                        maxlen=self.input_.shape[1], 
                                        padding='post')
        
        results = ''
        
        hidden = self.initialHiddenState(shape=(1, self.units))
        
        #TODO:
        testGraph = tf.Graph()
        with testGraph.as_default():
            x = tf.placeholder(dtype=tf.int32, name='input_',
                    shape=(self.batchSize, self.input_.shape[1]))
            h = tf.placeholder(dtype=tf.float32, name='hiddenstate',
                    shape=(self.batchSize, self.units)) 

            returenPredict = self.inference(x, None, h)
            
            saver = tf.train.Saver()
            latesetPath = tf.train.latest_checkpoint(modelPath)
				
            with tf.Session(graph = testGraph) as sess:
                print("Restoring from %s" %latesetPath)
                saver.restore(sess, latesetPath)
            
                sess.graph.finalize()
				
                predict = sess.run(returenPredict, 
                feed_dict={x: inputs, h: hidden})
									
                for pre in predict:
                    results += self.manLang.idx2word[pre] + ' '
                    if self.manLang.idx2word[pre] == '<end>':
                        break
                
        print(results)
						
				
    def gRU(self, units):
        return tf.keras.layers.GRU(units, 
                           return_sequences=True, 
                           return_state=True, 
                           recurrent_activation='sigmoid', 
                           recurrent_initializer='glorot_uniform')



    def initialHiddenState(self, shape):
        return np.zeros(shape)

#%%

def train():
    beginTime = time.time()
    
    fileName = r'./cmn-eng/cmn.txt'
    numSamples = 1000
    input_tensor, target_tensor, engLang, manLang, maxL_input, maxL_target =\
        load_dataset(fileName, numSamples)
        
    input_train, input_val, target_train, target_val =\
        train_test_split(input_tensor, target_tensor, test_size=0.2)
        
    trainModel = Model(numSamples=numSamples, batchSize=64, embeddingDim=256, 
                       units=1024, epochs=50, learningRate=1e-3, 
                       engLanguage=engLang, manLanguage=manLang,
                       input_=input_train, target=target_train,
                       ifTraining=True)
    
    savePath = r'./model/'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    trainModel.train(save_n=50, show_n=5, savePath=savePath)

    endTime = time.time()
    print("Totally spend %.2f seconds." %(endTime - beginTime))

    
def translate():    
    
    fileName = r'./cmn-eng/cmn.txt'
    numSamples = 1000
    input_tensor, target_tensor, engLang, manLang, maxL_input, maxL_target =\
        load_dataset(fileName, numSamples)
        
    input_train, input_val, target_train, target_val =\
        train_test_split(input_tensor, target_tensor, test_size=0.2)
        
    testModel = Model(numSamples=numSamples, batchSize=1, embeddingDim=256, 
                       units=1024, epochs=1, learningRate=1e-3, 
                       engLanguage=engLang, manLanguage=manLang,
                       input_=input_train, target=target_train,
                       ifTraining=False)
					   
    input_ = input("Please input the sentence: ")
    
    testModel.predict(sentence = input_, modelPath = r'./model/')
    

#%%
if __name__ == '__main__':
    #train()
    translate()
    
    
    