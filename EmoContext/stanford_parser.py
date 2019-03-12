#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 10:21
# @Author  : David
# @email   : mingren4792@126.com
# @File    : stanford_parser.py

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from nltk.tokenize import StanfordTokenizer

def stanford_tokenizer(str):

    tokenizer = StanfordTokenizer(path_to_jar='D:/software/stanford-parser-full-3.7/stanford-parser-3.7.0-models.jar')

    # sent = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."
    return tokenizer.tokenize(str)

# if __name__=='__main__':
#     sent = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."
#     result = stanford_tokenizer(sent)
#     print(result)


# st = StanfordPOSTagger('english-bidirectional-distsim.tagger')


# from nltk.tokenize import StanfordTokenizer
# s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."
# StanfordTokenizer().tokenize(s)
# s = "The colour of the wall is blue."
# StanfordTokenizer(options={"americanize": True}).tokenize(s)