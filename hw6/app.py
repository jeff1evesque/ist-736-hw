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
from config import quandl_api as q_creds
from consumer.twitter_query import TwitterQuery
from consumer.quandl_query import QuandlQuery
from view.exploratory import explore
from view.classifier import plot_bar
from exploratory.sentiment import Sentiment
from datetime import datetime
from controller.classifier import classify
import matplotlib.pyplot as plt

#
# local variables
#
data = {}
classify_results = {}
screen_name = [
    'jimcramer',
    'ReformedBroker',
    'TheStalwart',
    'LizAnnSonders',
    'SJosephBurns'
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

if not os.path.exists('data/quandl'):
    os.makedirs('data/quandl')


# instantiate api

t = TwitterQuery(
    t_creds['CONSUMER_KEY'],
    t_creds['CONSUMER_SECRET']
)
q = QuandlQuery(q_creds['API_KEY'])

#
# combine quandl with tweets
#
start_date = datetime(3000, 12, 25)
end_date = datetime(1000, 12, 25)

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
            data[sn] = t.query_user(
                sn,
                params=[
                    {'user': ['screen_name']},
                    'created_at',
                    'full_text',
                    {'retweeted_status': ['full_text']},
                    'retweet_count',
                    'favorite_count',
                    {'entities': ['user_mentions']}
                ],
                count=600,
                rate_limit=900
            )

            # sentiment analysis
            s = Sentiment(data[sn], 'full_text')
            data[sn] = pd.concat([s.vader_analysis(), data[sn]], axis=1)
            data[sn].replace('\s+', ' ', regex=True, inplace=True)

            # store locally
            data[sn].to_csv('data/twitter/{sn}.csv'.format(sn=sn))

        except Exception as e:
            print('Error: did not finish \'{sn}\'.'.format(sn=sn))
            print(e)

    # convert to string
    data[sn]['created_at'] = data[sn]['created_at'].astype(str)

    # largest time span
    start = data[screen_name[i]]['created_at'].iloc[0]
    temp_start = datetime.strptime(start.split()[0], '%Y-%m-%d')
    if temp_start < start_date:
        start_date = temp_start

    end = data[screen_name[i]]['created_at'].iloc[-1]
    temp_end = datetime.strptime(end.split()[0], '%Y-%m-%d')
    if temp_end > end_date:
        end_date = temp_end

#
# harvest quandl: using the maximum twitter date range
#
if Path('data/quandl/nasdaq.csv').is_file():
    df_nasdaq = pd.read_csv('data/quandl/nasdaq.csv')

else:
    df_nasdaq = q.get_ts(start_date=start_date, end_date=end_date)
    df_nasdaq.to_csv('data/quandl/nasdaq.csv')

#
# preprocess: left join on twitter dataset(s).
#
for i,sn in enumerate(screen_name):
    # merge with consistent date format
    data[sn]['created_at'] = [datetime.strptime(
        x.split()[0],
        '%Y-%m-%d'
    ) for x in data[sn]['created_at']]

    # convert to string
    data[sn]['created_at'] = data[sn]['created_at'].astype(str)
    data[sn]['full_text'] = data[sn]['full_text'].astype(str)

    #
    # some screen_name text multiple times a day, yet quandl only provides
    #     daily prices.
    #
    data[sn] = data[sn].groupby([
        'created_at',
        'screen_name'
    ]).agg({
        'full_text': lambda a: ''.join(a)
    }).reset_index()

    #
    # merge tweets with quandl
    #
    data[sn] = data[sn].join(
        df_nasdaq.set_index(['Trade Date']),
        how='left', on=['created_at']
    )

    #
    # merge days (weekend, holidays) with no ticker value to previous day.
    #
    drop_indices = []
    for i,row in data[sn].iterrows():
        if (i == 0 and np.isnan(data[sn]['Index Value'][i])):
            data[sn]['full_text'][i+1] = '{current} {next}'.format(
                current=data[sn]['full_text'][i],
                next=data[sn]['full_text'][i+1]
            )
            drop_indices.append(i)

        elif (i > 0 and not data[sn]['Index Value'][i-1]):
            continue

        elif (i > 0 and np.isnan(data[sn]['Index Value'][i])):
            if not np.isnan(data[sn]['Index Value'][i-1]):
                data[sn]['Index Value'][i] = data[sn]['Index Value'][i-1]
                data[sn]['High'][i] = data[sn]['High'][i-1]
                data[sn]['Low'][i] = data[sn]['Low'][i-1]
                data[sn]['Total Market Value'][i] = data[sn]['Total Market Value'][i-1]
                data[sn]['Dividend Market Value'][i] = data[sn]['Dividend Market Value'][i-1]
                data[sn]['full_text'][i] = '{previous} {current}'.format(
                    previous=data[sn]['full_text'][i-1],
                    current=data[sn]['full_text'][i-1]
                )
                drop_indices.append(i)

    #
    # drop rows: rows with no tickers and empty 'full_text'.
    #
    drop_indices.extend(data[sn][data[sn]['full_text'] == ''].index)
    data[sn] = data[sn].drop(data[sn].index[drop_indices]).reset_index()

    #
    # index data: relabel index as up (0) or down (1) based on previous time
    #
    data[sn]['trend'] = [0 if data[sn]['Index Value'][i] > data[sn]['Index Value'].get(i-1, 0)
        else 1
        for i,x in enumerate(data[sn]['Index Value'])]

    data[sn].to_csv('tests.csv')

    #
    # classify
    #
    classify_results[sn] = classify(
        data[sn],
        key_class='trend',
        key_text='full_text',
        directory='viz/{sn}'.format(sn=sn),
        top_words=25
    )

#
# ensembled scores
#
y_pos = np.arange(len(screen_name))
performance = [v[0] for k,v in classify_results.items()]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, screen_name)
plt.ylabel('Performance')
plt.savefig('viz/accuracy_overall.png')
plt.show()

#
# exploratory
#
df = pd.concat(data).reset_index()
explore(df, stopwords=stopwords, sent_cases={'screen_name': screen_name})
