#!/usr/bin/python

#
# this files the following packages:
#
#     pip install nltk
#     pip install -U scikit-learn
#     pip install scikit-plot
#

import time
import csv
import numpy as np
from pathlib import Path
import pandas as pd
from nltk import tokenize, download, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import scikitplot as skplt
download('vader_lexicon')


class Model():
    '''

    train classifier model using provided dataset.

    '''

    def __init__(
        self,
        df=None,
        vectorize=True,
        key_text='SentimentText',
        key_class='Sentiment',
        fp='{}/data/sample-sentiment.csv'.format(
            Path(__file__).resolve().parents[1]
        )
    ):
        '''

        define clasas variables.

        '''

        # class variables
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(fp)
        self.key_text = key_text
        self.key_class = key_class

        # vectorize data
        self.split()
        if vectorize:
            self.vectorize()

    def split(self, test_size=0.25):
        '''

        split data into train and test.

        '''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df[self.key_text],
            self.df[self.key_class],
            test_size=test_size
        )

    def get_split(self):
        '''

        return previous train and test split.

        '''

        return({
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
        })

    def get_pos(self, l):
        '''

        apply pos tagger to supplied list.

        '''

        return([pos_tag(x) for x in l])

    def vectorize(self):
        '''

        vectorize provided data.

        '''

        # bag of words: with 'english' stopwords
        self.count_vect = CountVectorizer(stop_words='english')
        bow = self.count_vect.fit_transform(self.X_train)

        # tfidf weighting
        self.tfidf_transformer = TfidfTransformer()
        self.X_train_tfidf = self.tfidf_transformer.fit_transform(bow)

    def get_tfidf(self):
        '''

        get current X_train tfidf.

        '''

        return(self.X_train_tfidf)

    def get_df(self):
        '''

        get original dataframe.

        '''

        return(self.df)

    def model(self, X, y, validate=False):
        '''

        create naive bayes model.

        @validate, must have tuple shape (X_test, y_test)

        '''

        # fit model
        self.clf = MultinomialNB().fit(X, y)

        # validate
        if validate and len(validate) == 2:
            predictions = []
            for item in list(validate[0]):
                prediction = self.count_vect.transform([item])
                predictions.append(
                    self.clf.predict(self.tfidf_transformer.fit_transform(prediction))
                )

            return({
                'model': self.clf,
                'actual': validate[1],
                'predicted': predictions
            })

        return({
            'model': self.clf,
            'actual': None,
            'predicted': None
        })

if __name__ == '__main__':
    #
    # unigram: perform unigram analysis.
    #
    unigram = Model()

    # unigram vectorize
    unigram_params = unigram.get_split()
    unigram_vectorized = unigram.get_tfidf()

    # unigram classifier
    model_unigram = unigram.model(
        unigram_vectorized,
        unigram_params['y_train'],
        validate=(unigram_params['X_test'], unigram_params['y_test'])
    )

    # plot unigram
    skplt.metrics.plot_confusion_matrix(
        model_unigram['actual'],
        model_unigram['predicted']
    )
    plt.show()

    # determine pos
    df_pos = unigram.get_df()
    df_pos['SentimentText'] = unigram.get_pos(
        df_pos['SentimentText'].apply(lambda x: x.split())
    )

    #
    # pos: perform part of speech analysis.
    #
    pos = Model(df=df_pos, vectorize=False)
    pos_params = pos.get_split()

    # pos classifier
    model_pos = pos.model(
        df_pos['SentimentText'],
        pos_params['y_train'],
        validate=(pos_params['X_test'], pos_params['y_test'])
    )

    # plot pos
    skplt.metrics.plot_confusion_matrix(
        model_pos['actual'],
        model_pos['predicted']
    )
    plt.show()
