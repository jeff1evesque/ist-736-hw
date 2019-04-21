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
        self.data = data
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
        for i, s in enumerate(self.data[self.column_name]):
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
        self.data['compound'] = result['compound']
        self.data['negative'] = result['negative']
        self.data['neutral'] = result['neutral']
        self.data['positive'] = result['positive']

        # return scores
        return(self.data)

    def plot_ts(self, title='Sentiment Analysis', filename='sentiment.png'):
        '''

        plot sentiment generated from 'vader_analysis'.

        '''

        # generate plot
        plt.figure()
        with pd.plotting.plot_params.use('x_compat', True):
            self.data.negative.plot(color='r', legend=True)
            self.data.positive.plot(color='g', legend=True)
            self.data.neutral.plot(color='b', legend=True)
        plt.title(title)

        # save plot
        plt.savefig(filename)
        plt.show()
