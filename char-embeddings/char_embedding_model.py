
import json, argparse, os
import re
import io
from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
from tool import ekphrasis_config
from capsule_net import Capsule
from Attention_layer import AttentionM
from keras.utils import to_categorical


from sklearn.svm import SVC
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, CuDNNGRU, Bidirectional, GRU, Input, Flatten, SpatialDropout1D, LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.models import load_model
from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
from tool import ekphrasis_config
from capsule_net import Capsule
from Attention_layer import AttentionM

from keras.models import Model
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, Input
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.decomposition import PCA
from keras.utils import plot_model
import numpy as np
import random
import sys
import csv
import os
import h5py
import time


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


def char_processing(trainTexts, name):
    for index in range(len(trainTexts)):
        if len(trainTexts[index]) >= 500:
            trainTexts[index] = trainTexts[index][:500]
    text = '\n '.join(trainTexts)
    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print('Vectorization...')
    X = np.zeros((len(trainTexts), 500), dtype=np.int)

    for i, sentence in enumerate(trainTexts):
        for t, char in enumerate(sentence):
            X[i, t] = char_indices[char]
    if name == 'train':
        return X, chars, char_indices
    else:
        return X

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
            if mode == "train" or mode == 'dev':
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
            string = re.sub("so+ ", ' so ', string)
            string = re.sub("lie+ ", ' lie ', string)
            string = re.sub("okay+ ", ' okay ', string)
            string = re.sub(' lol[a-z]+ ', ' laugh out loud ', string)
            string = re.sub(' wow+ ', ' wow ', string)
            string = re.sub('wha+ ', ' what ', string)
            string = re.sub(' ok[a-z]+ ', ' ok ', string)
            string = re.sub(' u+ ', ' you ', string)
            string = re.sub(' wellso+n ', ' well soon ', string)
            string = re.sub(' byy+ ', ' bye ', string.lower())
            string = re.sub(' ok+ ', ' ok ', string.lower())
            string = re.sub('o+h', ' oh ', string)
            string = re.sub('you+ ', ' you ', string)
            string = re.sub('plz+', ' please ', string.lower())
            string = string.replace('’', '\'').replace('"', ' ').replace("`", "'")
            string = string.replace('fuuuuuuukkkhhhhh', 'fuck')
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
            string = string.replace(" won ' t ", ' will not ').replace(' aint ', ' am not ')
            indices.append(int(line[0]))
            conversations.append(string.lower())
    if mode == "train" or mode == 'dev':
        return indices, conversations, labels
    else:
        return indices, conversations




embedding_dim = 300
batch_size = 128
use_pca = False
lr = 0.001
lr_decay = 1e-4
maxlen = 500



print("Processing training data...")
trainIndices, trainTexts, train_labels = preprocessData('./data/train.txt', mode="train")
train_labels = to_categorical(np.asarray(train_labels))
train_X, chars, char_indices = char_processing(trainTexts, name='train')

print("Processing dev data...")
devIndices, devTexts, dev_labels = preprocessData('./data/dev.txt', mode="dev")
dev_X = char_processing(trainTexts, name='dev')
dev_labels = to_categorical(np.asarray(dev_labels))

print("Processing test data...")
testIndices, testTexts = preprocessData('./data/testwithoutlabels.txt', mode="test")
test_X = char_processing(trainTexts, name='test')

embedding_vectors = {}
with open('./glove.840B.300d-char.txt', 'r') as f:
    for line in f:
        line_split = line.strip().split(" ")
        vec = np.array(line_split[1:], dtype=float)
        char = line_split[0]
        embedding_vectors[char] = vec

embedding_matrix = np.zeros((len(chars), 300))
# embedding_matrix = np.random.uniform(-1, 1, (len(chars), 300))
for char, i in char_indices.items():
    # print ("{}, {}".format(char, i))
    embedding_vector = embedding_vectors.get(char)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


Routings = 5
Num_capsule = 10
Dim_capsule = 32

embedding_layer = Embedding(
    len(chars), embedding_dim, input_length=maxlen,
    weights=[embedding_matrix])

# RNN Layer
sequence_input = Input(shape=(500,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
enc = Bidirectional(GRU(512, dropout=0.25, return_sequences=True))(embedded_sequences)
enc = Bidirectional(GRU(512, dropout=0.25, return_sequences=True))(enc)
att = AttentionM()(enc)
fc1 = Dense(128, activation="relu")(att)
fc2_dropout = Dropout(0.25)(fc1)
output = Dense(4, activation='sigmoid')(fc2_dropout)
model = Model(inputs=sequence_input, outputs=output)

rmsprop = optimizers.rmsprop(lr=0.003)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['acc'])
print(model.summary())
print(np.shape(train_X))
print(np.shape(train_labels))
model.fit(x=train_X, y=train_labels,
          batch_size=batch_size, epochs=25)

