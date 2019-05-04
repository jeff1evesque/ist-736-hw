#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from exploratory.sentiment import Sentiment
from exploratory.word_cloud import word_cloud

def explore(df):
    #
    # segregated data
    #
    df_lie_false = df.loc[df['lie'] == 0]
    df_lie_true = df.loc[df['lie'] == 1]
    df_sent_neg = df.loc[df['sentiment'] == 0]
    df_sent_pos = df.loc[df['sentiment'] == 1]

    #
    # word clouds
    #
    word_cloud(df_lie_false['review'], filename='viz/wc_lie_false.png')
    word_cloud(df_lie_true['review'], filename='viz/wc_lie_true.png')
    word_cloud(df_sent_neg['review'], filename='viz/wc_sentiment_negative.png')
    word_cloud(df_sent_pos['review'], filename='viz/wc_sentiment_positive.png')
    word_cloud(df['review'], filename='viz/wc_overall.png')

    #
    # sentiment analysis
    #
    sent_nolie = Sentiment(df_lie_false, 'review')
    sent_lie = Sentiment(df_lie_true, 'review')
    sent_neg = Sentiment(df_sent_neg, 'review')
    sent_pos = Sentiment(df_sent_pos, 'review')
    sent_overall = Sentiment(df, 'review')

    sent_nolie.vader_analysis()
    sent_lie.vader_analysis()
    sent_neg.vader_analysis()
    sent_pos.vader_analysis()
    sent_overall.vader_analysis()

    sent_nolie.plot_ts(title='No Lies', filename='viz/sentiment_no_lie.png')
    sent_lie.plot_ts(title='Lying Sentiment', filename='viz/sentiment_lie.png')
    sent_neg.plot_ts(title='Negative Sentiment', filename='viz/sentiment_neg.png')
    sent_pos.plot_ts(title='Positive Sentiment', filename='viz/sentiment_pos.png')
    sent_overall.plot_ts(title='Overall Sentiment', filename='viz/sentiment_overall.png')
