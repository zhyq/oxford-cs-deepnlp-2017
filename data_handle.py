import xml.etree.ElementTree as ET
import collections
import pickle
import numpy
class XmlHander():
    def __init__(self,xml_file):
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()

    def xml_trance(self):
        data={}

        #print(self.root.tag)
        for son in self.root:
            #print(child.tag)
            if son.tag != "file":
                continue
            talkid = ''
            title = ''
            description = ''
            keywords = ''
            content = ''
            for grandson in son:
                if grandson.tag == "head":
                    for node in grandson:
                        if node.tag == "talkid":
                            talkid = node.text
                        if node.tag == "title":
                            title = node.text
                        if node.tag == "description":
                            description = node.text
                        if node.tag == "keywords":
                            keywords = node.text
                if grandson.tag == "content":
                    content = grandson.text.strip()
            if talkid == '' :
                continue
            if talkid not in data.keys():
                data[talkid]={}
            data[talkid]['title'] = title
            data[talkid]['description'] = description
            data[talkid]['keywords'] = keywords
            data[talkid]['content'] = content
            data[talkid]['label'] = ['o','o','o']
            for keyword in keywords.split(','):
                if keyword.strip().lower() == 'technology':
                    data[talkid]['label'][0]='T'
                if keyword.strip().lower() == 'entertainment':
                    data[talkid]['label'][1]='E'
                if keyword.strip().lower() == 'design':
                    data[talkid]['label'][2]='D'
            data[talkid]['label'] = "".join(data[talkid]['label'])
        return data


class Vocab():
    def __init__(self):
        self.id_to_word=[]
        self.word_to_id=[]

    def build_vocab(self,words):
        data = words.replace("\n"," </s> ").split()
        counter = collections.Counter(data)
        pair_data = sorted(counter.items(),key=lambda x:(-x[1],x[0]))
        words,_ = list(zip(*pair_data))
        words = list(words)
        ### oov to zero
        words.insert(0,'<oov>')
        self.word_to_id = dict(zip(words,range(len(words))))
        self.id_to_word = dict(zip(range(len(words)),words))

    def word2id(self,words):
        ids = [self.word_to_id.get(word,0) for word in words.split()]
        return ids

    def id2word(self,ids):
        words = [self.id_to_word.get(id,'<oov>') for id in ids.split()]
        return " ".join(words)

    def save(self,path):
        with open(path,'wb') as f:
            pickle.dump(self.word_to_id,f)
            pickle.dump(self.id_to_word,f)

    def load(self,path):
        with open(path,'rb') as f:
            self.word_to_id = pickle.load(f)
            self.id_to_word = pickle.load(f)


def load_pretrained_wv(w2v_file,vocab_file):
    """
        input: w2v_file word2vec 模型
               vocab_file Vocab(word 和 id 对应关系的数据结构)加载后存储为pkl的文件
        return: vb_size:词典大小
                emb_size:词向量维度
                embd:array 加载的词向量 (词的vocab中id为array的小标)
    """
    vb = Vocab()
    vb.load(vocab_file)
    vb_size = len(vb.word_to_id.keys())

    f = open(w2v_file,'r')
    line = f.readline()
    _,emb_size = line.split(' ')
    emb_size = int(emb_size)

    embd = [[0]*emb_size]*vb_size
    for line in f.readlines():
        row = line.strip().split(' ')
        # oov
        idx=vb.word_to_id.get(row[0],emb_size)
        if idx != emb_size:
            embd[idx]=row[1:]
    print('Loaded word2vec!')
    f.close()
    return vb_size,emb_size,embd


class DataShuffle():
    def __init__(self,corpus,vocab_file,sentence_len=256):
        xh = XmlHander(corpus)
        data = xh.xml_trance()
        vb = Vocab()
        vb.load(vocab_file)
        self.labels=['ooo','Too','oEo','ooD','TEo','ToD','oED','TED']
        self.label_to_id = dict(zip(self.labels,range(len(self.labels))))
        self.id_to_label = dict(zip(range(len(self.labels)),self.labels))
        self.X = []
        self.y = []
        for talkid in data.keys():
            words = ""
            y=[0] * len(self.labels)
            words = words +" " + data[talkid]['title'].lower()
            words = words + " " + data[talkid]['description'].lower()
            words = words + " " + data[talkid]['content'].lower()
            words_id = vb.word2id(words)
            if len(words_id) < sentence_len:
                more_id = [0] * (sentence_len - len(words_id))
                words_id.extend(more_id)
            else:
                words_id = words_id[:sentence_len]
            labels_id = self.label_to_id.get(data[talkid]['label'],0)
            y[labels_id] = 1
            self.X.append(words_id)
            #self.y.append(labels_id)
            self.y.append(y)

        assert len(self.X) == len(self.y)



    def get_batch_data(self,batch_size=128):
        random_index = numpy.random.choice(len(self.X), batch_size, replace=True)
        X = []
        y = []
        for i in range(len(random_index)):
            X.append(self.X[random_index[i]])
            y.append(self.y[random_index[i]])

        X = numpy.array(X)
        y = numpy.array(y)
        return X,y








if __name__ == "__main__":
    #xh = XmlHander("data/ted_en-20160408_part.xml")
    xh = XmlHander("data/ted_en-20160408.xml")
    data = xh.xml_trance()
    words = ''
    for talkid in data.keys():
        words = words +" " + data[talkid]['title'].lower()
        words = words + " " + data[talkid]['description'].lower()
        words = words + " " + data[talkid]['content'].lower()
    vb = Vocab()
    vb.build_vocab(words)
    print(len(vb.word_to_id))
    vb.save("data/vocab.pkl")
