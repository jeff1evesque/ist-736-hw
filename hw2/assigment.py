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


class Ensemble():
    '''

    train classifier model using provided dataset.

    '''

    def __init__(self, fp='{}/data/sample-sentiment.csv'.format(
        Path(__file__).resolve().parents[1]
    ), key_text='SentimentText', key_class='Sentiment'):
        '''

        define clasas variables.

        '''

        # class variables
        self.data = pd.read_csv(fp)
        self.key_text = key_text
        self.key_class = key_class

        # vectorize data
        self.split()
        self.vectorize()

        # pos dataframe
        self.pos = self.data
        self.pos[self.key_text] = self.get_pos(
            self.data[self.key_text].apply(lambda x: x.split())
        )

    def split(self, test_size=0.25):
        '''

        split data into train and test.

        '''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data[self.key_text],
            self.data[self.key_class],
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

        apply stanford pos tagger to supplied list.

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

    def nb_model(self, X, y, validate=False):
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
    model = Ensemble()

    # naive bayes prediction
    model_params = model.get_split()
    vectorized = model.get_tfidf()

    # classifiers
    nb_unigram = model.nb_model(
        vectorized,
        model_params['y_train'],
        validate=(model_params['X_test'], model_params['y_test'])
    )

    skplt.metrics.plot_confusion_matrix(
        nb_unigram['actual'],
        nb_unigram['predicted']
    )
    plt.show()
