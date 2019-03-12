#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 10:28
# @Author  : David
# @email   : mingren4792@126.com
# @File    : statistic.py

import pandas as pd
import re


train = pd.read_table('./data/train.txt', sep='\t')
dev = pd.read_table('./data/devwithoutlabels.txt', sep='\t')


def statistic_item(data):
    others = 0
    angry = 0
    sad = 0
    happy = 0
    for row_num in range(len(data['id'])):
        if data['label'][row_num] == 'others':
            others += 1
        elif data['label'][row_num] == 'sad':
            sad += 1
        elif data['label'][row_num] == 'happy':
            happy += 1
        elif data['label'][row_num] == 'angry':
            angry += 1

    return others, angry, sad, happy


def merge_document(data, outputfile_name, label_flag):
    result = []
    for row_num in range(len(data)):
        line = []
        item = []
        if label_flag:
            item.append(data['label'][row_num])
        line.append(data['turn1'][row_num])
        line.append(data['turn2'][row_num])
        line.append(data['turn3'][row_num])
        str = ' '.join(line)
        item.append(str)
        result.append(item)
    if not label_flag:
        pf = pd.DataFrame(data=result, columns=['review'])
    else:
        pf = pd.DataFrame(data=result, columns=['label', 'review'])
    pf.to_csv('./data/version_1/'+ outputfile_name + '.csv', sep='\t', index=None, encoding='utf-8')


def caculat_acc():
    result = 0
    golden_label = pd.read_table('./data/dev.txt', sep='\t')
    pre = pd.read_table('./submit/test.txt', sep='\t')
    for row in range(len(pre['label'])):
        if golden_label['label'][row] == pre['label'][row]:
            result += 1
    print(round(result/len(pre['label']),6))



if __name__ == '__main__':
    # merge_document(train, 'train_merge', True)
    # str = "Damn it. It's so sad I think it\\'s ')'. Maybe. I think it\\'s ')'. Maybe."
    # str = str.replace("\\", '')
    # print(str)
    caculat_acc()