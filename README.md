# oxford-cs-deepnlp-2017_practical-2

### run
`python run.py`

### 说明
oxford的deepnlp的文本分类实验

* 1 加载预先训练好的词向量，采用加权平均作为每篇文章的词向量

* 2 采用softmax 分类。 多标签转为多分类问题 ('ooo','Too','oEo','ooD','TEo','ToD','oED','TED' 对应 1-8 8分类)

### 文件组成
`data` 数据文件 ted 的xml文件 和 基于 text8训练好的100维词向量
`model.py` softmax 分类模型文件
`run.py` 主流程文件
`data_helper.py` 数据处理文件 包括 word2vec模型加载 词和id相互转换 xml解析等

