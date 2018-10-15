import tensorflow as tf
import sklearn

class TextClass():
    def __init__(self,emb_size,vocab_size,sentence_len,class_num,batch_size):
        """

        """
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.sentence_len =sentence_len
        self.class_num = class_num
        self.batch_size = batch_size
        with tf.name_scope('input') as scope:
            self.input_x = tf.placeholder(tf.int32,[None,self.sentence_len],name="input_x")
            self.input_y = tf.placeholder(tf.int32,[None,self.class_num],name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
            self.emb_input = tf.placeholder(tf.float32,[self.vocab_size,self.emb_size],name="emb_input")

            self.emb = tf.Variable(tf.random_uniform(shape=[self.vocab_size,self.emb_size],minval=-1.0,maxval=1.0,dtype=tf.float32),trainable=True)
            #self.emb = tf.Variable(tf.random_uniform(shape=[self.vocab_size,self.emb_size],minval=-1.0,maxval=1.0,dtype=tf.float32),trainable=False)
            self.emb_init = self.emb.assign(self.emb_input)

        with tf.name_scope('layer'):
            # 线性和求平均
            self.x = tf.reduce_mean(tf.nn.embedding_lookup(self.emb,self.input_x),axis=1)

            self.W = tf.Variable(tf.random_uniform([self.emb_size,self.class_num],-1.0,1.0))
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
