# oxford-cs-deepnlp-2017_practical-2

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub issues](https://img.shields.io/github/issues/zhyq/oxford-cs-deepnlp-2017_practical-2.svg)](https://github.com/zhyq/oxford-cs-deepnlp-2017_practical-2/issues)
![GitHub stars](https://img.shields.io/github/stars/zhyq/oxford-cs-deepnlp-2017_practical-2.svg)

  
  oxford deepnlp practical-2 text class
  中文说明[中文](https://github.com/zhyq/oxford-cs-deepnlp-2017_practical-2/README_zh.cn)


### parctical description
   details[description](https://github.com/oxford-cs-deepnlp-2017/practical-2)

### Training
train run.py -m [basic or lstm or cnn] -l logdir

`python run.py -m baisc -l logs/basic_log ` basic model use pretrained word2vec   acc:99+%

`python run.py -m lstm -l logs/lstm_log `  lstm model acc:99.9%

`python run.py -m cnn -l logs/cnn_log`   cnn model acc:99.9%

`tensorboard -logdir logs/` tensorboard visualization 
 
 acc visualization
![image](https://raw.githubusercontent.com/zhyq/oxford-cs-deepnlp-2017_practical-2/master/png/acc.png)
 histogram
![image](https://raw.githubusercontent.com/zhyq/oxford-cs-deepnlp-2017_practical-2/master/png/histogram.png)


### model description

basic:model.py
* 1 load text8 pretrained word vectors，
* 2 softmax  ('ooo','Too','oEo','ooD','TEo','ToD','oED','TED' 8 classes)
![image](https://raw.githubusercontent.com/zhyq/oxford-cs-deepnlp-2017_practical-2/master/png/model.png)

lstm model:lstm.py
* 1 pretrained word vectors ,lstm 
* 2 softmax。
![image](https://raw.githubusercontent.com/zhyq/oxford-cs-deepnlp-2017_practical-2/master/png/lstm.png)

cnn model:cnn.py
* 1 pretrained word vectors ,cnn.
* 2 softmax
![image](https://raw.githubusercontent.com/zhyq/oxford-cs-deepnlp-2017_practical-2/master/png/cnn.png)

### file description
 
 * `data` data file : ted xml corpus , text8 ,word vectors
 * `model.py` basic model
 * `lstm.py`  lstm model
 * `cnn.py` cnn model
 * `run.py` main file
 * `data_helper.py` load xml file ... 

### Results
  
  * experiment 1 . basic model :word embedding 
     
     ```
      sentence_len=64 
      
      emb: text8 pretrained word embedding)
      
      x = average(emb(word1),emb(word2),...)   [emb(word1): word1 vector]
      
      y = softmax(wx+b)
      
      when: emb trainable=False
      
      acc :73%
      ```

  * 2 experiment 2. basic model  word embedding
      
      ```
      when: emb trainable=True
      
      acc: 99%
      ```

  * experiment 3. lstm model
      
      ```
      x: [x = matrix(emb(word1),emb(word2)...)]
      
      x' = lstm(x)   [lstm output]
      
      y = softmax(wx'+b)
      
      acc:99.9%
      ```

  * experiment 4. cnn model
      
      ```
      x: input  [x = matrix(emb(word1),emb(word2)...)]
      
      x' = cnn(x)   
      
      y = softmax(wx'+b)
      
      acc:99.9%
      ```

   * 5 conclusion
  
       in this practical . cnn > lstm > basic model

20180724 add basic model

20181013 add lstm model

20181015 add cnn model



