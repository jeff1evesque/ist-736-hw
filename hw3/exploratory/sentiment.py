#!/usr/bin/python

# this file the following packages:
#
#     pip install nltk
#     pip install vaderSentiment
#

from nltk import download
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
download('vader_lexicon')


class Sentiment():
    '''

    use vader to perform sentiment analyis.

    '''

    def __init__(self, data, column_name):
        '''

        define class variables.

        '''

        # local variables
        self.df = data
        self.column_name = column_name

    def vader_analysis(self):
        '''

        perform sentiment analysis.

        '''

        analyser = SentimentIntensityAnalyzer()

        sid = SentimentIntensityAnalyzer()
        result = {
            'compound': [],
            'negative': [],
            'neutral': [],
            'positive': []
        }

        # sentiment analysis
        for i, s in enumerate(self.df[self.column_name]):
            ss = sid.polarity_scores(s)

            for k in sorted(ss):
                if k == 'compound':
                    result['compound'].append(ss[k])
                elif k == 'neg':
                    result['negative'].append(ss[k])
                elif k == 'neu':
                    result['neutral'].append(ss[k])
                elif k == 'pos':
                    result['positive'].append(ss[k])

        # append results
        self.df['compound'] = result['compound']
        self.df['negative'] = result['negative']
        self.df['neutral'] = result['neutral']
        self.df['positive'] = result['positive']

        # return scores
        return(self.df)

    def plot_ts(self, title='Sentiment Analysis', filename='sentiment.png'):
        '''

        plot sentiment generated from 'vader_analysis'.

        '''

        # generate plot
        plt.figure()
        with pd.plotting.plot_params.use('x_compat', True):
            self.df.negative.plot(color='r', legend=True)
            self.df.positive.plot(color='g', legend=True)
            self.df.neutral.plot(color='b', legend=True)
        plt.title(title)

        # save plot
        plt.savefig(filename)
        plt.show()
