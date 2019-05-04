#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython
#

import os
import re
from consumer.twitter_query import TwitterQuery
from config import twitter_api as creds
from pathlib import Path
import pandas as pd

# local variables
csv_taralipinski = 'data/twitter/taralipinski.csv'

# create directories
if not os.path.exists('data/twitter'):
    os.makedirs('data/twitter')

if not os.path.exists('viz'):
    os.makedirs('viz')

# instantiate api
q = TwitterQuery(
    creds['CONSUMER_KEY'],
    creds['CONSUMER_SECRET']
)

#
# tweets: taralipinski
#
if Path(csv_taralipinski).is_file():
    df_taralipinski = pd.read_csv(csv_taralipinski)

else:
    df_taralipinski = q.query_user('taralipinski', count=6000, rate_limit=900)
    df_taralipinski['text'] = df_taralipinski['text'].str.replace(
        'http\S+|www.\S+',
        '',
        case=False
    )
    df_taralipinski.to_csv(csv_taralipinski)
