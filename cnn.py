import tensorflow as tf
import sklearn
from functools import reduce
class TextClass():
    def __init__(self,emb_size,vocab_size,sentence_len,class_num,batch_size):
        """

        """
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.sentence_len =sentence_len
        self.class_num = class_num
        self.batch_size = batch_size
        with tf.name_scope("input"):
            self.input_x = tf.placeholder(tf.int32,[None,self.sentence_len],name="input_x")
            self.input_y = tf.placeholder(tf.int32,[None,self.class_num],name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")

            self.emb = tf.Variable(tf.random_uniform(shape=[self.vocab_size,self.emb_size],minval=-1.0,maxval=1.0,dtype=tf.float32),trainable=True)
            #self.emb = tf.Variable(tf.random_uniform(shape=[self.vocab_size,self.emb_size],minval=-1.0,maxval=1.0,dtype=tf.float32),trainable=False)
            self.emb_input = tf.placeholder(tf.float32,[self.vocab_size,self.emb_size],name="emb_input")
            self.emb_init = self.emb.assign(self.emb_input)

        with tf.name_scope('layer'):
            self.x_emb = tf.nn.embedding_lookup(self.emb,self.input_x)

            # cnn
            #卷积和高度 为 3 4 5 ，对于文本卷积，卷积的宽度为emb_size
            filter_sizes = [3,4,5]
            #卷积的通道数 决定这卷积的输出大小
            num_filters = [128,128,128]
            num_filter_total = reduce(lambda x,y:x+y,num_filters)
            self.x = self.cnn(self.x_emb,filter_sizes,num_filters,num_filter_total)
            self.W = tf.Variable(tf.random_uniform([num_filter_total,self.class_num],-1.0,1.0))
            self.b = tf.Variable(tf.zeros(self.class_num))
            self.y = tf.matmul(self.x,self.W) + self.b
            tf.summary.histogram('layer/weights', self.W)
            tf.summary.histogram('layer/bias', self.b)
            tf.summary.histogram('layer/output', self.y)

        with tf.name_scope('optimizer'):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.y))
            self.train_step = tf.train.GradientDescentOptimizer(0.05).minimize(self.cross_entropy)
        with tf.name_scope('loss'):
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            tf.summary.scalar('acc', self.accuracy)
            self.merged = tf.summary.merge_all()

    def cnn(self,x,filter_sizes,num_filters,num_filter_total):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        def max_pool(x,filter_size):
            return tf.nn.max_pool(x, ksize=[1, self.sentence_len - filter_size + 1, 1, 1],strides=[1, 1, 1, 1], padding='VALID')

        #x_image = tf.reshape(x,[-1,self.sentence_len,self.emb_size,1])
        x_image = tf.expand_dims(x,-1)

        pooled_outputs = []

        for filter_size,num_filter in zip(filter_sizes,num_filters):
            W_conv = weight_variable([filter_size, self.emb_size, 1, num_filter])
            b_conv = bias_variable([num_filter])
            h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
            h_pool = max_pool(h_conv,filter_size)
            pooled_outputs.append(h_pool)

        h_pool = tf.concat(pooled_outputs,axis=3)
        h_pool_flat = tf.reshape(h_pool,[-1,num_filter_total])
        h_pool_dropout = tf.nn.dropout(h_pool_flat,self.dropout_keep_prob)
        return h_pool_dropout
