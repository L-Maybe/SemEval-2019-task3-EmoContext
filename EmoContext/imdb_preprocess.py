# -*- coding: utf-8 -*-
# @Time    : 2018/4/13 14:32
# @Author  : David
# @Email   : mingren4792@126.com
# @File    : mywassa_2018_cnn.py
# @Software: PyCharm

import numpy as np
import re
import os
import pickle
import gensim
import pandas as pd
import io

from collections import defaultdict
from config import *

embedding_dim = 200

train = pd.read_csv('./data/train.csv', sep='\t', quoting=3)
dev = pd.read_csv('./data/test.csv', sep='\t', quoting=3)

# train["label"] = train["label"] .replace({'happy': 0,
#                                             'angry': 1, 'sad': 2,
#                                             'others': 3
#                                             })
# dev["label"] = dev["label"].replace({'happy': 0,
# #                                             'angry': 1, 'sad': 2,
# #                                             'others': 3
# #                                             })


def clean_str(string):
    """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
    string = string.replace(':)', ' smile ').replace(':-)', ' smile ') \
        .replace(':D', ' smile ').replace('=)', ' smile ').replace('😄', ' smile ').replace('☺', ' smile ')
    string = string.replace('❤', ' like ').replace('<3', ' like ').replace('💕', ' like ').replace('😍', ' like ')
    string = string.replace('🤗', ' happy ').replace(':-)', ' happy ')
    string = string.replace(':(', ' unhappy ').replace(':-(', ' unhappy ').replace('💔', ' unhappy ') \
        .replace('😕', 'unhappy ').replace('😤', ' unhappy ')
    string = string.replace('😡', ' anger ').replace('🙃', ' anger ')
    string = string.replace('😞', ' sadness ').replace('😓', ' sadness ').replace('😔', ' sadness ')
    string = string.replace(';-;', ' unhappy ')

    string = string.replace('’', '\'').replace('"',' ')
    string = string.replace('whats ', 'what is')
    string = string.replace('Iam ', 'I am').replace(' iam ', 'i am').replace(' dnt ', ' do not ')
    string = string.replace('I ve ', 'I have ').replace('I m ', 'I\'am ').replace('i m ', 'i\'m ')
    string = string.replace('Iam ', 'I am ').replace('iam ', 'i am ')
    string = string.replace('dont ', 'do not ').replace('google.co.in ', 'google').replace('hve ', 'have ')
    string = string.replace(' F ', ' Fuck ').replace('Ain\'t ', ' are not ').replace(' lv ', ' love ')
    string = string.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', 'It is').replace(' its ', ' it is ')
    string = string.replace('  Nd  ', ' and ').replace(' nd ', ' and ')
    string = string.replace('Thnx ', ' Thanx ').replace('[#TRIGGERWORD#]', '')

    # delete ascll
    string = re.sub('[^\x00-\x7f]', ' ', string)
    word1_list = string.split()
    for index in range(len(word1_list)):
        if word1_list[index] in LOGOGRAM.keys():
            word1_list[index] = LOGOGRAM[word1_list[index]]
    string = ' '.join(word1_list)
    # letters only
    # string = re.sub("[^a-zA-Z\'.!?]", " ", string)
    string = string.lower()
    word_list = string.split()
    for index in range(len(word_list)):
        if word_list[index] in LOGOGRAM.keys():
            word_list[index] = LOGOGRAM[word_list[index]]

    string = " ".join(word_list)

    # words = stanford_tokenizer(string)

    # stops = set(stopwords.words("english"))
    # meaningful_words = [w for w in words if not w in stops]
    # string = " ".join(words)
    return string


def tranform_label_to_num(sentiment):
    if 'happy'== sentiment:
        return 0
    elif 'angry'== sentiment:
        return 1
    elif 'sad' == sentiment:
        return 2
    elif 'others' == sentiment:
        return 3


def build_data_train(train_data, test_data, clean_string=True, train_ratio=0.9):
    """
    Loads data and split into train and test sets.
    """

    revs = []
    vocab = defaultdict(float)

    # Pre-process train data set

    for i in range(len(train_data['review'])):
        print('第%d条， 共%d条' % (i, len(train_data['review'])))
        rev = train_data['review'][i]
        y = train_data['label'][i]
        if clean_string:
            orig_rev = clean_str(rev)
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        #  计算某个单词在文本中出现的次数
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': 1}
        revs.append(datum)

    for i in range(len(test_data['review'])):
        print('第%d条， 共%d条' % (i, len(test_data['review'])))
        rev = test_data['review'][i]
        if clean_string:
            orig_rev = clean_str(rev)
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': 0}
        revs.append(datum)




    # for i in range(len(trial_data['review'])):
    #     print('第%d条， 共%d条' %(i, len(trial_data['review'])))
    #     rev = trial_data['review'][i]
    #     y = tranform_label_to_num(trial_labels.values[i][0])
    #     if clean_string:
    #         orig_rev = clean_str(rev)
    #     else:
    #         orig_rev = ' '.join(rev).lower()
    #     words = set(orig_rev.split())
    #     for word in words:
    #         vocab[word] += 1
    #     datum = {'y': y,
    #              'text': orig_rev,
    #              'num_words': len(orig_rev.split()),
    #              'split': -1}
    #     revs.append(datum)





    return revs, vocab


def load_bin_vec(model, vocab, k=embedding_dim):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1

    return word_vecs, unk_words


def get_W(word_vecs, k=embedding_dim):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((embedding_dim, ))
    W[1] = np.random.uniform(-0.25, 0.25, k)
    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]

        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map


if __name__ == '__main__':
    revs, vocab = build_data_train(train, dev, clean_string=True)
    w2vec_file = 'F:/vector/glove_model.txt'
    max_l = np.max(pd.DataFrame(revs)['num_words'])

    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join('/home','lidawei','vector', 'glove.twitter.27B.200d.txt'), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    # model = KeyedVectors.load_word2vec_format(w2vec_file, binary=True, encoding='utf-8')
    # model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=True)

    # model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=False)
    #  w2v, unk_words = load_bin_vec(model, vocab)

    w2v, unk_words = load_bin_vec(embeddingsIndex, vocab)

    W, word_idx_map = get_W(w2v)
    data_processed = os.path.join('pickle', 'emoContext.pickle')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(data_processed, 'wb'))

