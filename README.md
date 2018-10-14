# oxford-cs-deepnlp-2017_practical-2
  oxford的deepnlp的文本分类实验
  采用不同方法实现，对比效果分析

### 实验说明
   见课程实验要求[实验描述](https://github.com/oxford-cs-deepnlp-2017/practical-2)

### 如何运行
`python run.py -m baisc` 使用预先训练的word2vec词向量做线性平均加之后，做线性分类

`python run.py -m lstm`  使用预先训练的word2vec词向量，采用lstm深度学习模型分类

### 说明

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

### 实验组效果对比
  
  * 1 实验一
     (```)
      sentence_len=64 
      emb: text8预先训练好的词向量模型)
      x = average(emb(word1),emb(word2),...)   [emb(word1):表示 word1的词向量,这里的x就是文章词向量的线性加平均]
      y = softmax(wx+b)
      当 emb 不可参与训练时(trainable=False)
      acc :73%
      (```)
  * 2 实验二
      (```)
      条件同1 不过 emb可以参与训练优化
      acc: 99%
      (```)
  * 3 实验三
      (```)
      x: 用词向量矩阵表示 [x = matrix(emb(word1),emb(word2)...)]
      x' = lstm(x)   [对输入的x做lstm层转换]
      y = softmax(wx'+b)
      无论 emb 是否参与训练优化
      acc:99.9%
      (```)
   * 4 结论
   此实验中，仅仅只训练 w b两个参数是不够的。这就是deep 多参数的好处

未完，后续同步深度学习方法
20180724 实现basic模型
20181013 add 增加lstm模型



