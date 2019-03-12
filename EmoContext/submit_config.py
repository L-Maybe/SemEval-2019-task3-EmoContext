#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 22:32
# @Author  : David
# @email   : mingren4792@126.com
# @File    : submit_config.py

import pandas as pd

dev_data = pd.read_table('./data/devwithoutlabels.txt', sep='\t')
dev_pre = pd.read_csv('./result/emoContext_pre.csv')
dev_pre["label"] = dev_pre["label"] .replace({0:"others", 1:"happy", 2: "sad", 3:"angry"})

result = []
for row in range(len(dev_pre)):
    line = []
    line.append(dev_data['id'][row])
    line.append(dev_data['turn1'][row])
    line.append(dev_data['turn2'][row])
    line.append(dev_data['turn3'][row])
    line.append(dev_pre['label'][row])
    result.append(line)

pf = pd.DataFrame(data=result, columns=['id', 'turn1', 'turn2', 'turn3', 'label'])
pf.to_csv('./submit/test.txt', sep='\t', index=False, encoding='utf8 ')