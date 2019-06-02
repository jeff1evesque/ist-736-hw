#!/usr/bin/python

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from exploratory.sentiment import Sentiment
from exploratory.word_cloud import word_cloud
from utility.dataframe import cleanse

def explore(df, sent_cases=None, stopwords=[], target='full_text', suffix=''):
    '''

    generate wordclouds and sentiment series plot.

    @target, dataframe column to parse
    @sent_cases, dict where keys represent columns, and values represent list
        of possible column values. This is used to filter the dataframe.

    '''

    if suffix:
        suffix = '{suffix}_'.format(suffix=suffix)

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
                    filename='viz/{value}/wc{suffix}.png'.format(
                        value=v,
                        suffix=suffix
                    ),
                    stopwords=stopwords
                )

                # create sentiment plot
                sent_temp = Sentiment(wc_temp, target)
                sent_temp.vader_analysis()
                sent_temp.plot_ts(
                    title='{value}'.format(value=v),
                    filename='viz/{value}/sentiment{suffix}.png'.format(
                        value=v,
                        suffix=suffix
                    )
                )

            word_cloud(
                df[target],
                filename='viz/wc_overall{suffix}.png'.format(suffix=suffix)
            )
            sent_overall = Sentiment(df, target)
            sent_overall.vader_analysis()
            sent_overall.plot_ts(
                title='Overall Sentiment',
                filename='viz/sentiment_overall{suffix}.png'.format(
                    suffix=suffix
                )
            )

    else:
        # clean text
        wc_temp[target] = cleanse(df, target, ascii=True)

        # create wordcloud
        word_cloud(
            df[target],
            filename='viz/wc{suffix}.png'.format(suffix=suffix),
            stopwords=stopwords
        )

        # create sentiment plot
        sent_temp = Sentiment(df, target)
        sent_temp.vader_analysis()
        sent_temp.plot_ts(
            title='Sentiment Analysis',
            filename='viz/sentiment{suffix}.png'.format(suffix=suffix)
        )
