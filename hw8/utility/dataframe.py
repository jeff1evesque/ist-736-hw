#!/usr/bin/python

import re
import string
import pandas as pd

def standardize_df(fp, delimiter=','):
    '''

    generate dataframe from csv containing unbalanced columns.

    '''

    # local variables
    largest_column_count = 0

    # determine max column
    with open(fp, 'r') as f:
        lines = f.readlines()

        for l in lines:
            column_count = len(l.split(delimiter)) + 1

            if largest_column_count < column_count:
                largest_column_count = column_count

    # close file
    f.close()

    # temporary dataframe: allows uneven columns
    column_names = ['col-{}'.format(i) for i in range(largest_column_count)]

    #
    # remove first row: will generally have the following form:
    #
    #     0,column-1,column-2,column-n nan nan nan nan nan nan nan
    #
    df = pd.read_csv(fp, header=None, delimiter=delimiter, names=column_names)
    return(df.iloc[1:])

def cleanse(df, column, ascii=False):
    '''

    remove twitter account, punctuations, urls, lowercase.

    @string.punctuation, '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

    '''

    if ascii:
        pattern_ascii = r'[^\x00-\x7f]'
    else:
        pattern_ascii = ''

    pattern_twitter_act = '@[a-zA-Z0-9_]{0,15}'
    pattern_url = 'https?://[A-Za-z0-9./]+'
    pattern_newline=r'[\\n|\\r\\n]'
    pattern_punctuation = '[{p}]'.format(p=string.punctuation)
    pattern = '|'.join((
        pattern_twitter_act,
        pattern_url,
        pattern_newline,
        pattern_punctuation,
        pattern_ascii
    ))
    r = re.compile(pattern)

    return([re.sub(r, '', str(w)) for w in df[column]])
