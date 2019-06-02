#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython Quandl wordcloud scikit-plot
#

import os
import sys
import math
import pandas as pd
from lxml import etree as et
from pathlib import Path
from controller.topic_model import topic_model
sys.path.append('..')

#
# local variables
#
df = None
dir = '../data/110'

#
# create directories
#
if not os.path.exists('viz'):
    os.makedirs('viz')

if not os.path.exists('data'):
    os.makedirs('data')

#
# create dataframe
#
if (
    Path('data/topic_model_1.csv').is_file() and
    Path('data/topic_model_2.csv').is_file()
):
    df = pd.concat([
        pd.read_csv('data/topic_model_1.csv'),
        pd.read_csv('data/topic_model_2.csv')
    ])

else:
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            dtype = os.path.basename(subdir).split('-')[-2:]
            fp = os.path.join(subdir, file)

            if Path(fp).is_file():
                file = Path(fp)

                with open(fp, 'rb') as f:
                    data = '<data>{content}</data>'.format(
                        content=f.read()
                    )

                parser = et.XMLParser(recover=True)
                tree = et.fromstring(data, parser=parser)

                df_cols = ['docno', 'text', 'gender', 'party']
                if df is None:
                    df = pd.DataFrame(columns = df_cols)

                for doc in tree.getchildren():
                    tm_docno, tm_text = None, None
                    for el in doc.getchildren():
                        if el is not None:
                            if el.tag == 'DOCNO':
                                tm_docno = el.text if el.text else None
                            elif el.tag == 'TEXT':
                                tm_text = el.text if el.text else None

                    df = df.append(
                        pd.Series(
                            [tm_docno, tm_text, dtype[0], dtype[1]], 
                            index = df_cols
                        ), 
                        ignore_index=True
                    )

    #
    # split dataframes: too big for github
    #
    row_count = df.shape[0]
    split_index = math.ceil(row_count / 2)

    df.iloc[:split_index,:].to_csv('data/topic_model_1.csv')
    df.iloc[split_index:,:].to_csv('data/topic_model_2.csv')

#
# topic modeling
#
topic_model(df, rotation=0)
