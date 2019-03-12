#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 10:31
# @Author  : David
# @email   : mingren4792@126.com
# @File    : demo.py

import gensim
import re
import pandas as pd
import random as rd

from stanford_parser import stanford_tokenizer
from config import LOGOGRAM
from collections import defaultdict


def vector_demo():
    w2vec_file = 'F:/vector/glove_model.txt'
    model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=False)
    print(model["Yup"])


def logogram_processing(review):
    string = review.replace('’', '\'')
    string = string.replace('Iam ', ' I am').replace(' iam ', ' i am').replace(' dnt ', ' do not ')
    string = string.replace('I ve ', ' I have ').replace('I m ', ' I\'am ').replace('i m ', ' i\'m ')
    string = string.replace('Iam ', ' I am ').replace('iam ', ' i am ')
    string = string.replace('dont ', ' do not ').replace('google.co.in ', 'google').replace('hve ', ' have ')
    string = string.replace(' F ', ' Fuck ').replace('Ain\'t ', ' can\'t ').replace(' lv ', ' love ')
    string = string.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', ' It is ').replace(' its ', ' it is ')
    string = string.replace('  Nd  ', ' and ').replace(' nd ', ' and ')
    string = string.replace('Thnx ', 'Than')
    review_list = string.split(' ')
    for index in range(len(review_list)):
        if review_list[index] in LOGOGRAM.keys():
            review_list[index] = LOGOGRAM[review_list[index]]
    print(' '.join(review_list))

    return ' '.join(review_list)



def divided_train_data():
    data = pd.read_csv('./data/version_1/train_merge.csv', sep='\t')
    test = pd.read_csv('./data/version_1/dev_merge.csv', sep='\t')

    train_data = []
    dev_data = []
    test_data = []
    for row in range(24000):
        line = []
        print('train:%d', row)
        line.append(logogram_processing(data['review'][row]))
        line.append(data['label'][row])
        train_data.append(line)

    for row in range(24000,len(data)):
        line = []
        print('train:%d', row)
        line.append(logogram_processing(data['review'][row]))
        line.append(data['label'][row])
        dev_data.append(line)

    for row in range(len(test)):
        line = []
        print('test:%d', row)
        line.append(logogram_processing(data['review'][row]))
        line.append('others')
        test_data.append(line)
    train_data_result = pd.DataFrame(data=train_data)
    train_data_result.to_csv('./data/bert_version/train.tsv', sep='\t', index=None, header=None)

    dev_data_result = pd.DataFrame(data=dev_data)
    dev_data_result.to_csv('./data/bert_version/dev.tsv', sep='\t', index=None, header=None)

    dev_data_result = pd.DataFrame(data=test_data)
    dev_data_result.to_csv('./data/bert_version/test.tsv', sep='\t', index=None, header=None)


def get_max_len():
    max = 0
    data = pd.read_csv('./data/version_1/train_merge.csv', sep='\t')
    for row in range(len(data)):
        if max < len(data['review'][row].split()):
            max = len(data['review'][row].split())
    print(max)


def segment(text):
    from ekphrasis.classes.tokenizer import SocialTokenizer
    social_tokenizer = SocialTokenizer(lowercase=True).tokenize
    return ' '.join(social_tokenizer(text))


def data_process():
    data = pd.read_csv('./data/version_1/train_merge.csv', sep='\t')
    result = []
    for row in range(1000):
        line = []
        text = data['review'][row]
        line.append(data['label'][row])
        line.append(' '.join(stanford_tokenizer(text)))
        result.append(line)

    df = pd.DataFrame(result, columns=['label', 'review'])
    df.to_csv('./data/version_2_ekphtasis/demo.csv', sep='\t', index=False)


def statistic_emotext():
    vocab = defaultdict(float)
    data = pd.read_csv('./data/version_1/train_merge.csv', sep='\t')
    for row in range(len(data)):
        vocab[data['label'][row]] += 1

    print(vocab)


def augment_data():
    data = pd.read_csv('./data/wassa/wassa_bert.csv', sep='\t')
    emotext_data = pd.read_csv('./data/version_1/train_merge.csv', sep='\t')
    angry = 0
    sad = 0
    happy = 0
    result = []

    for line_num in range(len(emotext_data)):
        line = []
        line.append(emotext_data['label'][line_num])
        line.append(emotext_data['review'][line_num])
        result.append(line)

    for row in range(len(data)):
        if angry == 9500:
            break
        line = []
        if data['label'][row] == 'angry':
            angry += 1
            line.append(data['label'][row])
            line.append(data['review'][row])
            result.append(line)

    for row in range(len(data)):
        if sad == 9600:
            break
        line = []
        if data['label'][row] == 'sad':
            sad += 1
            line.append(data['label'][row])
            line.append(data['review'][row])
            result.append(line)


    for row in range(len(data)):
        if happy == 10800:
            break
        line = []
        if data['label'][row] == 'happy':
            happy += 1
            line.append(data['label'][row])
            line.append(data['review'][row])
            result.append(line)


    df = pd.DataFrame(result, columns=['label', 'review']).sample(frac=1).reset_index(drop=True)
    df.to_csv('./data/version_2_augment/augment.csv', sep='\t', index=False)


def downsampling():
    # 减少others类别的数据总数，以达到数据平衡
    data = pd.read_csv('./data/version_1/train_merge.csv', sep='\t')
    result = []
    others = 0
    for row in range(len(data['label'])):
        line = []
        if data['label'][row] == 'others' and others <= 5000:
            others += 1
            line.append(data['label'][row])
            line.append(data['review'][row])
            result.append(line)

        elif data['label'][row] in ['sad', 'happy', 'angry']:
            line.append(data['label'][row])
            line.append(data['review'][row])
            result.append(line)
        else:
            continue
    df = pd.DataFrame(result, columns=['label', 'review'])
    df.to_csv('./data/downsampling/train.csv', sep='\t', index=False)


def upsampling():
    # 过采样；将数据少的部分进行拷贝以达到数据平衡的目的
    result = []
    data = pd.read_csv('./data/version_1/train_merge.csv', sep='\t')
    for row in range(len(data)):
        line = []
        if data['label'][row] == 'others':
            line.append(data['review'][row])
            line.append(data['label'][row])
            result.append(line)
        else:
            for i in range(3):
                line = []
                line.append(data['review'][row])
                line.append(data['label'][row])
                result.append(line)
    df = pd.DataFrame(result, columns=['review', 'label']).sample(frac=1).reset_index(drop=True)
    df.to_csv('./data/upsampling/upsampling_augment.csv', index=False, sep='\t')


def split_augment():
    data = pd.read_csv('./data/downsampling/train.csv', sep='\t')
    train_result = []
    dev_result = []
    for row in range(len(data)):
        line = []
        string = data['review'][row]
        string = clean_str(string)
        line.append(data['label'][row])
        line.append(string)
        if rd.random() < 0.8:
            train_result.append(line)
        else:
            dev_result.append(line)

    df = pd.DataFrame(train_result, columns=['label', 'review']).sample(frac=1).reset_index(drop=True)
    df.to_csv('./data/downsampling/train.tsv', index=False, sep='\t')

    df = pd.DataFrame(dev_result, columns=['label', 'review']).sample(frac=1).reset_index(drop=True)
    df.to_csv('./data/downsampling/dev.tsv', index=False, sep='\t')




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


split_augment()
