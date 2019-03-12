#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/1/12 10:45 
# @Author : DAVID 
# @Mail ： mingren4792@126.com
# @Site :  
# @File : tool.py 
# @Software: PyCharm

from ekphrasis.classes.tokenizer import SocialTokenizer



def ekphrasis_config(str):
    social_tokenizer = SocialTokenizer(lowercase=False).tokenize
    return social_tokenizer(str)


