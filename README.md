# oxford-cs-deepnlp-2017_practical-2
  oxford的deepnlp的文本分类实验
  采用向量平均加 lstm cnn 三种方法实现，对比效果分析

### 实验说明
   见课程实验要求[实验描述](https://github.com/oxford-cs-deepnlp-2017/practical-2)

### 如何运行
`python run.py -m baisc` 基本模型 使用预先训练的word2vec词向量做线性加平均作为输入进行分类 acc:99+%

`python run.py -m lstm`  lstm模型 使用预先训练的word2vec词向量，采用lstm深度学习模型分类 acc:99.9%

`python run.py -m cnn`   cnn模型 使用预先训练的word2vec词向量，采用cnn模型分类 acc:99.9%
### 说明

基本模型:model.py
* 1 加载使用text8语料预先训练好的词向量，采用加权平均作为每篇文章的词向量
* 2 采用softmax 分类。 多标签转为多分类问题 ('ooo','Too','oEo','ooD','TEo','ToD','oED','TED' 对应 1-8 8分类)

lstm模型:lstm.py
* 1 采用text8语料预先训练好的词向量，然后用lstm做特征筛选
* 2 采用softmax分类。

cnn模型:cnn.py
* 1 采用text8训练好的词向量，然后用卷积三层卷积
* 2 采用softmax分类
### 文件组成
 
 * `data` 数据文件 ted 的xml文件 和 基于 text8训练好的100维词向量
 * `model.py` word2vec词向量平均和作为特征做分类模型
 * `lstm.py`  lstm模型
 * `cnn.py` cnn模型
 * `run.py` 主流程文件
 * `data_helper.py` 数据处理文件 包括 word2vec模型加载 词和id相互转换 xml解析等

### 实验组效果对比
  
  * 1 实验一 基本模型，word embedding 不优化
     
     ```
      sentence_len=64 
      
      emb: text8预先训练好的词向量模型)
      
      x = average(emb(word1),emb(word2),...)   [emb(word1):表示 word1的词向量,这里的x就是文章词向量的线性加平均]
      
      y = softmax(wx+b)
      
      当 emb 不可参与训练时(trainable=False)
      
      acc :73%
      ```

  * 2 实验二 基本模型 word embedding 优化
      
      ```
      条件同1 不过 emb可以参与训练优化
      
      acc: 99%
      ```

  * 3 实验三 lstm模型
      
      ```
      x: 用词向量矩阵表示 [x = matrix(emb(word1),emb(word2)...)]
      
      x' = lstm(x)   [对输入的x做lstm层转换]
      
      y = softmax(wx'+b)
      
      无论 emb 是否参与训练优化
      
      acc:99.9%
      ```

  * 4 实验四 cnn 模型
      
      ```
      x: 用词向量矩阵表示 [x = matrix(emb(word1),emb(word2)...)]
      
      x' = cnn(x)   [对输入的x做了高度分为为 3 4 5 的三层卷积转换，然后concat作为特征输出]
      
      y = softmax(wx'+b)
      
      无论 emb 是否参与训练优化
      
      acc:99.9%


   * 5 结论
  
      此实验中，由实验一看出，仅仅只训练 w b两个参数是不够的。这就是deep 多参数的好处
       
      根据实验要求，cnn模型比lstm更合适，因为cnn可以从文本中卷积出关键词更加有利于文本的分类。
      而lstm在语言的前后影响方面更优，比如分词 实体识别等

未完，后续同步深度学习方法

20180724 实现basic模型

20181013 add 增加lstm模型

20181015 add 增加cnn模型



