#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython
#

import os
import sys
import re
import numpy as np
sys.path.append('..')
from consumer.twitter_query import TwitterQuery
from exploratory.word_cloud import word_cloud
from config import twitter_api as creds
from pathlib import Path
import pandas as pd
from exploratory.sentiment import Sentiment
from model.naive_bayes import Model as nb
import matplotlib.pyplot as plt

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

#
# generate wordcloud
#
if not os.path.exists('viz'):
    os.makedirs('viz')
###word_cloud(df_cnn['text'], filename='viz/wc_cnn.png')
###word_cloud(df_foxnews['text'], filename='viz/wc_foxnews.png')
###word_cloud(df['text'], filename='viz/wc_cnn_foxnews.png')

#
# sentiment analysis
#
sent_cnn = Sentiment(df_cnn, 'text')
sent_foxnews = Sentiment(df_foxnews, 'text')
sent_overall = Sentiment(df, 'text')

df_cnn = sent_cnn.vader_analysis()
df_foxnews = sent_foxnews.vader_analysis()
df_overall = sent_overall.vader_analysis()

# vectorize 'screen_name'
df_overall = df_overall.replace({'screen_name': {'CNN': 0, 'FoxNews': 1}})

###sent_cnn.plot_ts(title='CNN Sentiment', filename='viz/sentiment_cnn.png')
###sent_foxnews.plot_ts(title='FoxNews Sentiment', filename='viz/sentiment_foxnews.png')
###sent_overall.plot_ts(title='Overall Sentiment', filename='viz/sentiment_overall.png')

#
# classifier: use naive bayes to predict cnn and foxnews tweets.
#

# unigram: perform unigram analysis.
unigram = nb(df=df, key_text='text', key_class='screen_name')

# vectorize data
vectorized = unigram.get_tfidf()
unigram.split()
params = unigram.get_split()

# train classifier
unigram.model(
    params['X_train'],
    params['y_train'],
    validate=(params['X_test'], params['y_test'])
)

# plot unigram
unigram.plot_cm(filename='viz/cm_unigram.png')

# perform pos analysis
df_pos = unigram.get_df()

# reduce to ascii
regex = r'[^\x00-\x7f]'
df_pos['pos'] = [re.sub(regex, r' ', sent).split() for sent in df_pos['text']]

# suffix pos
df_pos['pos'] = [unigram.get_pos(x) for x in df_pos['pos']]

#
# new dataframe
#
# @pos_split, appends pos to word before vectorization and tfidf.
#
pos = nb(df_pos, key_text='pos', key_class='screen_name', lowercase=False)

# pos vectorize
pos_vectorized = pos.get_tfidf()
pos.split(pos_split=True)
pos_params = pos.get_split()

# pos classifier
model_pos = pos.model(
    pos_params['X_train'],
    pos_params['y_train'],
    validate=(pos_params['X_test'], pos_params['y_test'])
)

# plot pos
pos.plot_cm(filename='viz/cm_pos.png')

# ensembled scored
score_unigram = unigram.get_accuracy()
score_pos = pos.get_accuracy()
score_good = (score_unigram + score_pos) / 2
score_bad = 1 - score_good


objects = ('unigram', 'pos', 'overall')
y_pos = np.arange(len(objects))
performance = [score_unigram, score_pos, score_good]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance')
plt.savefig('viz/accuracy_overall.png')
plt.show()
