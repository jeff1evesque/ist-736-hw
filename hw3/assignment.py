#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython
#

import os
import sys
sys.path.append('..')
from consumer.twitter_query import TwitterQuery
from exploratory.word_cloud import word_cloud
from config import twitter_api as creds
from pathlib import Path
import pandas as pd

# local variables
csv_cnn = '../data/tweets_cnn.csv'
csv_foxnews = '../data/tweets_foxnews.csv'
q = TwitterQuery(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

# tweets: cnn
if Path(csv_cnn).is_file():
    df_cnn = pd.read_csv(csv_cnn)
else:
    df_cnn = q.query_user('cnn')
    df_cnn['text'] = df_cnn['text'].str.replace(
        'http\S+|www.\S+',
        '',
        case=False
    )
    df_cnn.to_csv(csv_cnn)

# tweets: foxnews
if Path(csv_foxnews).is_file():
    df_foxnews = pd.read_csv(csv_foxnews)
else:
    df_foxnews = q.query_user('FoxNews')
    df_foxnews['text'] = df_foxnews['text'].str.replace(
        'http\S+|www.\S+',
        '',
        case=False
    )
    df_foxnews.to_csv(csv_foxnews)

# combine dataframes
df = df_cnn.append(df_foxnews)

# generate wordcloud
if not os.path.exists('viz'):
    os.makedirs('viz')
word_cloud(df_cnn['text'], filename='viz/wc_cnn.png')
word_cloud(df_foxnews['text'], filename='viz/wc_foxnews.png')
word_cloud(df['text'], filename='viz/wc_cnn_foxnews.png')
