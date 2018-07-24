from model import *
from  data_handle import *
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
if __name__ == "__main__":
    #xh = XmlHander("data/ted_zh-cn-20160408.xml")
    # 训练和预测的预料
    corpus = "data/ted_en-20160408.xml"
    # word 和 id的相互映射关系文件
    vocab_file = "data/vocab.pkl"
    # 预先训练好的词向量模型 使用预先训练的词向量的好处 1 防止train预料不足导致预测 oov 2 词向量初始化更加科学
    w2v_model = "data/vectors.txt"

    # 1 加载 ted 的xml 数据 data
    """
    logging.info("load xml data")
    xh = XmlHander(corpus)
    data = xh.xml_trance()
    words = ''
    for talkid in data.keys():
        words = words +" " + data[talkid]['title'].lower()
        words = words + " " + data[talkid]['description'].lower()
        words = words + " " + data[talkid]['content'].lower()
    """
    # 2 建立 词 和 id的映射关系 vocab
    logging.info("build vocab")
    vb = Vocab()
    #vb.build_vocab(words)
    #vb.save(vocab_file)
    vb.load(vocab_file)
    # 3 加载预先训练的词向量 word embedding
    logging.info("load w2v model")
    vb_size,emb_size,embd = load_pretrained_wv(w2v_model,vocab_file)

    # 4 初始化 model 模型
    logging.info("init model")
    tc = TextClass(emb_size=emb_size, vocab_size=vb_size, sentence_len=256, class_num=8)

    # 5 session
    logging.info("init session")
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # 6 把 word2vec 模型加载到 model的emb中
    logging.info("init model emb")
    feed_dict = {
    	tc.emb_input : embd
    }
    session.run([tc.emb_init],feed_dict = feed_dict)

    # 7 train
    logging.info("train")
    ds = DataShuffle(corpus,vocab_file,sentence_len=256)
    for step in range(10000):
        X,y =  ds.get_batch_data()
        feed_dict = {
                tc.input_x : X,
                tc.input_y : y,
                tc.dropout_keep_prob : 0.8
                }

        logging.debug("X shape ")
        logging.debug(X.shape)
        logging.debug("y shape ")
        logging.debug(y.shape)
        logging.debug(X)
        logging.debug(y)
        train_step,acc = session.run([tc.train_step,tc.accuracy],feed_dict=feed_dict)
        tcx,tcy = session.run([tc.x,tc.y],feed_dict=feed_dict)
        logging.debug("tcx shape ")
        logging.debug(tcx.shape)
        logging.debug("tcy shape ")
        logging.debug(tcy.shape)
        logging.debug(tcx)
        logging.debug(tcy)
        #logging.info(y)
        #logging.info(tcy)
        #logging.info("-------")
        if step % 200 == 0:
            logging.info("step %d,acc:%.3lf" % (step,acc))






