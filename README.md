# oxford-cs-deepnlp-2017_practical-2

### run
`python run.py -m baisc` 使用预先训练的word2vec词向量做线性平均加之后，做线性分类

`python run.py -m lstm`  使用预先训练的word2vec词向量，采用lstm深度学习模型分类

### 说明
oxford的deepnlp的文本分类实验

基本模型:model.py
* 1 加载使用text8语料预先训练好的词向量，采用加权平均作为每篇文章的词向量
* 2 采用softmax 分类。 多标签转为多分类问题 ('ooo','Too','oEo','ooD','TEo','ToD','oED','TED' 对应 1-8 8分类)
lstm模型:lstm.py
* 1 采用text8语料预先训练好的词向量，然后用lstm做特征帅选
* 2 采用softmax分类。

### 文件组成
 
 * `data` 数据文件 ted 的xml文件 和 基于 text8训练好的100维词向量
 * `model.py` word2vec词向量平均和作为特征做分类模型
 * `lstm.py`  lstm模型
 * `run.py` 主流程文件
 * `data_helper.py` 数据处理文件 包括 word2vec模型加载 词和id相互转换 xml解析等

未完，后续同步深度学习方法
20180724 实现basic模型
20181013 add 增加lstm模型



