#!/usr/bin/python

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from exploratory.sentiment import Sentiment
from exploratory.word_cloud import word_cloud
from utility.dataframe import cleanse

def explore(df, sent_cases=None, stopwords=[], target='full_text'):
    '''

    generate wordclouds and sentiment series plot.

    @target, dataframe column to parse
    @sent_cases, dict where keys represent columns, and values represent list
        of possible column values. This is used to filter the dataframe.

    '''

    if sent_cases:
        cases = []
        for k,val in sent_cases.items():
            for v in val:
                # load select data
                wc_temp = df.loc[df[k] == v]

                # clean text
                wc_temp[target] = cleanse(wc_temp, target, ascii=True)

                #
                # create directories
                #
                if not os.path.exists('viz/{value}'.format(value=v)):
                    os.makedirs('viz/{value}'.format(value=v))

                # create wordcloud
                word_cloud(
                    wc_temp[target],
                    filename='viz/{value}/wc.png'.format(value=v),
                    stopwords=stopwords
                )

                # create sentiment plot
                sent_temp = Sentiment(wc_temp, target)
                sent_temp.vader_analysis()
                sent_temp.plot_ts(
                    title='{value}'.format(value=v),
                    filename='viz/{value}/sentiment.png'.format(value=v)
                )

            word_cloud(df[target], filename='viz/wc_overall.png')
            sent_overall = Sentiment(df, target)
            sent_overall.vader_analysis()
            sent_overall.plot_ts(
                title='Overall Sentiment',
                filename='viz/sentiment_overall.png'
            )

    else:
        # clean text
        wc_temp[target] = cleanse(df, target, ascii=True)

        # create wordcloud
        word_cloud(
            df[target],
            filename='viz/wc.png',
            stopwords=stopwords
        )

        # create sentiment plot
        sent_temp = Sentiment(df, target)
        sent_temp.vader_analysis()
        sent_temp.plot_ts(
            title='Sentiment Analysis',
            filename='viz/sentiment.png'
        )
