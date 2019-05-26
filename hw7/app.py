#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython Quandl wordcloud scikit-plot
#

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
from config import twitter_api as t_creds
from consumer.twitter_query import TwitterQuery
from view.exploratory import explore
from view.classifier import plot_bar
from exploratory.sentiment import Sentiment
from datetime import datetime
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
    if Path('data/twitter/{sn}.csv'.format(sn=sn)).is_file():
        data[sn] = pd.read_csv('data/twitter/{sn}.csv'.format(sn=sn))

    else:
        try:
            data[sn] = t.query(query=sn, count=600, rate_limit=900)
            data[sn].to_csv('data/twitter/{sn}.csv'.format(sn=sn))

        except Exception as e:
            print('Error: did not finish \'{sn}\'.'.format(sn=sn))
            print(e)
