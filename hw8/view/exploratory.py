#!/usr/bin/python

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from exploratory.sentiment import Sentiment
from exploratory.word_cloud import word_cloud
from utility.dataframe import cleanse

def explore(
    df,
    sent_cases=None,
    stopwords=[],
    target='full_text',
    background_color='white',
    suffix='',
    plot_wc=True,
    plot_sentiment=True,
    plot_wc_overall=True,
    plot_sentiment_overall=True,
    cleanse=True
):
    '''

    generate wordclouds and sentiment series plot.

    @target, dataframe column to parse
    @sent_cases, dict where keys represent columns, and values represent list
        of possible column values. This is used to filter the dataframe.

    '''

    if sent_cases:
        for k,val in sent_cases.items():
            for v in val:
                if plot_wc:
                    if cleanse:
                        wc_temp = df.loc[df[k] == v]
                        wc_temp[target] = cleanse(wc_temp, target, ascii=True)

                    # create wordcloud
                    word_cloud(
                        wc_temp,
                        filename='viz/wc_{value}{suffix}.png'.format(
                            value=v,
                            suffix=suffix
                        ),
                        stopwords=stopwords,
                        background_color=background_color
                    )

                if plot_sentiment:
                    # create sentiment plot
                    sent_temp = Sentiment(wc_temp, target)
                    sent_temp.vader_analysis()
                    sent_temp.plot_ts(
                        title='{value}'.format(value=v),
                        filename='viz/sentiment_{value}{suffix}.png'.format(
                            value=v,
                            suffix=suffix
                        )
                    )

        if plot_wc_overall:
            word_cloud(
                df[target],
                filename='viz/wc_overall{suffix}.png'.format(suffix=suffix),
                stopwords=stopwords,
                background_color=background_color
            )

        if plot_sentiment_overall:
            sent_overall = Sentiment(df, target)
            sent_overall.vader_analysis()
            sent_overall.plot_ts(
                title='Overall Sentiment',
                filename='viz/sentiment_overall{suffix}.png'.format(
                    suffix=suffix
                )
            )

    else:
        if plot_wc:
            # clean text
            wc_temp = df
            wc_temp[target] = [' '.join(x) for x in wc_temp[target]]
            wc_temp[target] = cleanse(df, target, ascii=True)

            # create wordcloud
            word_cloud(
                wc_temp,
                filename='viz/wc{suffix}.png'.format(suffix=suffix),
                stopwords=stopwords,
                background_color=background_color
            )

        if plot_sentiment:
            # create sentiment plot
            sent_temp = Sentiment(wc_temp, target)
            sent_temp.vader_analysis()
            sent_temp.plot_ts(
                title='Sentiment Analysis',
                filename='viz/sentiment{suffix}.png'.format(suffix=suffix)
            )
