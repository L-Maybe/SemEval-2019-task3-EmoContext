import os

import pandas as pd
import codecs
import re

# input_file = os.path.join('corpus', 'train.csv')
input_file = os.path.join('data', 'train.csv')
# output_file = os.path.join('output', 'train.arff')
output_file = os.path.join('affective_tweet', 'transform', 'train.arff')

input_df = pd.read_table(input_file, sep='\t', quoting=3)
input_df = input_df['review']

def to_arff(data_frame, output_file):
    with codecs.open(output_file, 'w', 'utf8') as my_file:
        header='@relation '+ input_file +'\n\n@attribute sentence string\n\n@data\n'
        my_file.write(header)

        for i in range(len(data_frame)):
            text = data_frame[i]
            out_line = '\'' + text.replace("\\", '').replace('\'', r'\'')+ "\'\n"

            my_file.write(out_line)

to_arff(input_df, output_file)