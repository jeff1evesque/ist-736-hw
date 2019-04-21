#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython
#

import sys
sys.path.append('..')
from consumer.twitter_query import TwitterQuery
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
    df_cnn.to_csv(csv_cnn)

# tweets: foxnews
if Path(csv_foxnews).is_file():
    df_foxnews = pd.read_csv(csv_foxnews)
else:
    df_foxnews = q.query_user('FoxNews')
    df_foxnews.to_csv(csv_foxnews)

# combine dataframes
df = df_cnn.append(df_foxnews)

print(df.head())
