#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython Quandl wordcloud scikit-plot
#

import os
import scipy.stats
from pathlib import Path
import pandas as pd
import numpy as np
from config import twitter_api as t_creds
from consumer.twitter_query import TwitterQuery
from view.exploratory import explore
from view.classifier import plot_bar
from exploratory.sentiment import Sentiment
from controller.classifier import classify
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

#
# local variables
#
data = {}
screen_name = [
    'GameofThrones',
    'GameofThronesFinale',
    'GOTFinale'
]
stopwords=[
    'http',
    'https',
    'nhttps',
    'RT',
    'amp',
    'co',
    'TheStreet'
]
stopwords.extend(screen_name)

#
# create directories
#
if not os.path.exists('data/twitter'):
    os.makedirs('data/twitter')

if not os.path.exists('../data/twitter'):
    os.makedirs('../data/twitter')

if not os.path.exists('viz'):
    os.makedirs('viz')

# instantiate api
t = TwitterQuery(
    t_creds['CONSUMER_KEY'],
    t_creds['CONSUMER_SECRET']
)

#
# classify
#
for i,sn in enumerate(screen_name):
    #
    # create directories
    #
    if not os.path.exists('viz/{sn}'.format(sn=sn)):
        os.makedirs('viz/{sn}'.format(sn=sn))

    #
    # harvest tweets
    #
    if Path('../data/twitter/{sn}.csv'.format(sn=sn)).is_file():
        data[sn] = pd.read_csv('../data/twitter/{sn}.csv'.format(sn=sn))

    else:
        try:
            data[sn] = t.query(query=sn, count=600, rate_limit=15)
            data[sn].to_csv('../data/twitter/{sn}.csv'.format(sn=sn))

        except Exception as e:
            print('Error: did not finish \'{sn}\'.'.format(sn=sn))
            print(e)

# combine samples
if Path('data/twitter/sample.csv').is_file():
    df_sample = pd.read_csv('data/twitter/sample.csv')

else:
    df_sample = [data[x].sample(500) for x in [*data]]
    df_sample = pd.concat(df).reset_index()
    df_sample.drop(['index', 'Unnamed: 0'], axis=1, inplace=True)
    df_sample.to_csv('data/twitter/sample.csv')

# load mturk
if Path('../data/mturk/got.csv').is_file():
    df_mturk = pd.read_csv('../data/mturk/got.csv')

#
# aggregate dataframe: get most frequent label across tweets, select first
#     instance between ties.
#
df = df_mturk.groupby(['Input.index', 'Input.full_text'])['Answer.sentiment.label'].agg(
    lambda x: scipy.stats.mode(x)[0][0]
)
