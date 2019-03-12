# SemEval-2019-task3-EmoContext


The code of SemEval-2019 task3 EmoContext: Multi-Step Ensemble Neural Network for Sentiment Analysis in Textual Conversation

The training of Elmo word vectors used in EmoContext is in the Elmo_pre_train file, where elmo_result.py is the training code. There is no sentence training, but word-by-word training, which lacks contextual information about words.

Training of word vectors in EmoContext in char-embeddings.

Instructions are available in the Elmo Pre-training and Word Vector Training folders. Check it out yourself. And I have written the sample code.

ELMO pre-training and word vector training use other people's projects, please pay attention to the source when using.

## The system architecture.

![image](https://github.com/L-Maybe/SemEval-2019-task3-EmoContext/blob/master/Architecture.png)

## The result of system.

![image](https://github.com/L-Maybe/SemEval-2019-task3-EmoContext/blob/master/result.png)


## enviroment
* Win 10
* tensorflow-gpu == 1.8.0
* keras == 2.2.4
* gensim == 3.6.0