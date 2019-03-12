"""
Example from training to saving.
"""
import argparse
import os
import re
import numpy as np
import io
import pandas as pd

from anago.utils import load_data_and_labels, load_glove, filter_embeddings
from anago.preprocessing import ELMoTransformer, IndexTransformer
from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
from tool import ekphrasis_config
from collections import defaultdict


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '
                line = cSpace.join(lineSplit)

            emoji_repeatedChars = TWEMOJI_LIST
            for emoji_meta in emoji_repeatedChars:
                emoji_lineSplit = line.split(emoji_meta)
                while True:
                    try:
                        emoji_lineSplit.remove('')
                        emoji_lineSplit.remove(' ')
                        emoji_lineSplit.remove('  ')
                        emoji_lineSplit = [x for x in emoji_lineSplit if x != '']
                    except:
                        break
                emoji_cSpace = ' ' + TWEMOJI[emoji_meta][0] + ' '
                line = emoji_cSpace.join(emoji_lineSplit)

            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)

            conv = ' <eos> '.join(line[1:4]) + ' '

            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)

            string = re.sub("tha+nks ", ' thanks ', conv.lower())
            string = re.sub("Tha+nks ", ' Thanks ', string.lower())
            string = re.sub("yes+ ", ' yes ', string.lower())
            string = re.sub("Yes+ ", ' Yes ', string)
            string = re.sub("very+ ", ' very ', string)
            string = re.sub("go+d ", ' good ', string)
            string = re.sub("Very+ ", ' Very ', string)
            string = re.sub("why+ ", ' why ', string.lower())
            string = re.sub("wha+t ", ' what ', string)
            string = re.sub("sil+y ", ' silly ', string)
            string = re.sub("hm+ ", ' hmm ', string)
            string = re.sub(" no+ ", ' no ', ' '+string)
            string = re.sub("sor+y ", ' sorry ', string)
            string =re.sub('sorry+', ' sorry ', string)
            string = re.sub("so+ ", ' so ', string)
            string = re.sub("lie+ ", ' lie ', string)
            string = re.sub("okay+ ", ' okay ', string)
            string = re.sub(' lol[a-z]+ ', ' laugh out loud ', string)
            string = re.sub(' wow+ ', ' wow ', string)
            string = re.sub(' wha+ ', ' what ', string)
            string = re.sub(' ok[a-z]+ ', ' ok ', string)
            string = re.sub(' u+ ', ' you ', string)
            string = re.sub(' wellso+n ', ' well soon ', string)
            string = re.sub(' byy+ ', ' bye ', string.lower())
            string = re.sub(' bye+', 'bye', string.lower())
            string = re.sub(' ok+ ', ' ok ', string.lower())
            string = re.sub(' o+h', ' oh ', string)
            string = re.sub(' you+ ', ' you ', string)
            string = re.sub(' plz+', ' please ', string.lower())
            string = re.sub('  off+ ', ' off ', string.lower())
            string = re.sub('jokes+', ' joke ', string.lower())
            string = re.sub('hey+', ' hey ', string.lower())
            string = re.sub('will+', ' will ', string.lower())
            string = re.sub('right+', 'tight', string.lower())
            string = string.replace('’', '\'').replace('"', ' ').replace("`", "'")
            string = string.replace('fuuuuuuukkkhhhhh', ' fuck ').replace('f*ck ', ' fuck ')
            string = string.replace('fuckk', ' fuck ').replace('donot', ' do not ')
            string = string.replace('whats ', 'what is ').replace("what's ", 'what is ').replace("i'm ", 'i am ')
            string = string.replace("it's ", 'it is ')
            string = string.replace('Iam ', 'I am ').replace(' iam ', ' i am ').replace(' dnt ', ' do not ')
            string = string.replace('I ve ', 'I have ').replace('I m ', ' I am ').replace('i m ', 'i am ')
            string = string.replace('Iam ', 'I am ').replace('iam ', 'i am ')
            string = string.replace('dont ', 'do not ').replace('google.co.in ', ' google ').replace(' hve ', ' have ')
            string = string.replace('Ain\'t ', ' are not ').replace(' lv ', ' love ')
            string = string.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', ' It is').replace(' its ', ' it is ')
            string = string.replace('  Nd  ', ' and ').replace(' nd ', ' and ').replace('i ll ', 'i will ')
            string = string.replace(" I'd ", ' i would ').replace('&apos;', "'")
            string = ' ' + string.lower()
            for item in LOGOGRAM.keys():
                string = string.replace(' ' + item + ' ', ' ' + LOGOGRAM[item].lower() + ' ')

            list_str = ekphrasis_config(string)
            for index in range(len(list_str)):
                if list_str[index] in EMOTICONS_TOKEN.keys():
                    list_str[index] = EMOTICONS_TOKEN[list_str[index]][1:len(EMOTICONS_TOKEN[list_str[index]]) - 1].lower()

            for index in range(len(list_str)):
                if list_str[index] in LOGOGRAM.keys():
                    list_str[index] = LOGOGRAM[list_str[index]].lower()

            for index in range(len(list_str)):
                if list_str[index] in LOGOGRAM.keys():
                    list_str[index] = LOGOGRAM[list_str[index]].lower()

            string = ' '.join(list_str)
            string = string.replace(" won ' t ", ' will not ').replace(' aint ', ' am not ').replace('#','')
            string = string.replace('*', '').replace('nooo', 'no').replace(' umder stand ', 'understand')
            string = string.replace('justsaying', ' just saying ').replace('whos ', 'who is')
            indices.append(int(line[0]))
            conversations.append(string.lower())
    if mode == "train":
        return indices, conversations, labels
    else:
        return indices, conversations

def get_word_dict():
    trainDataPath = './data/train.txt'
    devDataPath = './data/devwithoutlabels.txt'
    testDataPath = './data/testwithoutlabels.txt'
    word_map = defaultdict(float)
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
    devIndices, devTexts = preprocessData(devDataPath, mode="dev")
    testIndices, testTexts = preprocessData(testDataPath, mode="test")
    for train_item in trainTexts:
        train_text_list = train_item.split()
        for train_word in train_text_list:
            word_map[train_word] += 1

    for test_item in testTexts:
        test_text_list = test_item.split()
        for test_word in test_text_list:
            word_map[test_word] += 1

    for dev_item in devTexts:
        dev_text_list = dev_item.split()
        for test_word in dev_text_list:
            word_map[test_word] += 1

    return word_map


def elmo_feature():
    word_map = get_word_dict()
    result = []
    print('Transforming datasets...')
    p = ELMoTransformer()
    i = 0
    for item in word_map.keys():
        i += 1
        print(i)
        vec = []
        train = [[item+'']]
        word_vec = p.transform(train)
        vec.append(item)
        meta_vec = word_vec[0][0][0]
        meta_vec = meta_vec.tolist()
        vec.append(meta_vec)
        result.append(vec)
    df = pd.DataFrame(data=result)
    df.to_csv('./vec/elmo.csv', sep='\t', index=False, header=None, encoding='utf-8')


def char_featur():
    word_map = get_word_dict()
    result = []
    c = IndexTransformer()
    for item in word_map.keys():
        vec = []
        train = [[item + '']]
        word_vec = c.transform(train)
        print(word_vec)
        vec.append(item)
        meta_vec = word_vec[0][0][0]
        meta_vec = meta_vec.tolist()
        vec.append(meta_vec)
        result.append(vec)
        print(result)





if __name__ == '__main__':
    elmo_feature()
