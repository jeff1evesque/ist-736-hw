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
import re
import string
import numpy as np
from pathlib import Path
import pandas as pd
from nltk.corpus import stopwords
from nltk import tokenize, download, pos_tag
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import scikitplot as skplt
from model.penn_treebank import penn_scale
stop_words = set(stopwords.words('english'))
download('vader_lexicon')


class Model():
    '''

    train classifier model using provided dataset.

    '''

    def __init__(
        self,
        df=None,
        vectorize=True,
        stem=True,
        lowercase=True,
        key_text='SentimentText',
        key_class='Sentiment',
        fp='{}/data/sample-sentiment.csv'.format(
            Path(__file__).resolve().parents[1]
        )
    ):
        '''

        define class variables.

        '''

        # class variables
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(fp)

        self.key_text = key_text
        self.key_class = key_class
        self.actual = None
        self.predicted = None

        #
        # clean: remove twitter account, punctuations and urls, lowercase,
        #        stem each word.
        #
        # @string.punctuation, '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        #
        pattern_twitter_act = '@[a-zA-Z0-9_]{0,15}'
        pattern_url = 'https?://[A-Za-z0-9./]+'
        pattern_punctuation = '[{p}]'.format(p=string.punctuation)
        pattern = '|'.join((pattern_twitter_act, pattern_url, pattern_punctuation))

        self.df[self.key_text] = [re.sub(pattern, '', w) for w in self.df[self.key_text]]

        if lowercase:
            self.df[self.key_text] = [w.lower() for w in self.df[self.key_text]]

        if stem:
            p = PorterStemmer()
            self.df[self.key_text] = self.df[self.key_text].apply(
                lambda x: [' '.join([p.stem(w) for w in x.split(' ')])][0]
            )

        # vectorize data
        if vectorize:
            self.vectorize()
            self.split()

    def split(self, test_size=0.20, pos_split=False):
        '''

        split data into train and test.

        '''

        # split
        if pos_split:
            for i, row in self.df.iterrows():
                # max length
                if isinstance(self.df[self.key_text].iloc[i], str):
                    max_length = len(self.df[self.key_text].iloc[i].split())
                else:
                    max_length = len(self.df[self.key_text].iloc[i].str.split())
                pos = self.df[['pos']].iloc[i]

                # rebuild 'key-text' with pos suffix
                combined = ''
                for j in range(max_length):
                    if isinstance(self.df[self.key_text][i], str):
                        word = self.df[self.key_text][i].split()[j]
                    else:
                        word = self.df[self.key_text].iloc[i].split()[j]

                    combined = '{combined} {word}-{pos}'.format(
                        combined=combined,
                        word=word,
                        pos=pos[0][j]
                    )
                self.df[self.key_text].iloc[[i]] = combined

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.vectorize(self.df[self.key_text])[1],
                self.df[self.key_class],
                test_size=test_size
            )

        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.tfidf,
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

        pos = pos_tag(l)
        result = ' '.join(['{word}-{pos}'.format(
            word=l[i],
            pos=penn_scale[x[1]]
        ) if x[1] in penn_scale else '{word}-{pos}'.format(
            word=l[i],
            pos=1
        ) for i,x in enumerate(pos)])
        return(result)

    def vectorize(self, data=None, stop_words='english'):
        '''

        vectorize provided data.

        '''

        if data is not None:
            # bag of words: with 'english' stopwords
            count_vect = CountVectorizer(stop_words=stop_words)
            bow = count_vect.fit_transform(data)

            # tfidf weighting
            tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
            tfidf = tfidf_vectorizer.fit_transform(data)

            return(bow, tfidf)

        else:
            # bag of words: with 'english' stopwords
            self.count_vect = CountVectorizer(stop_words=stop_words)
            self.bow = self.count_vect.fit_transform(self.df[self.key_text])

            # tfidf weighting
            self.tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
            self.tfidf = self.tfidf_vectorizer.fit_transform(self.df[self.key_text])

    def get_tfidf(self):
        '''

        get current X_train tfidf.

        '''

        return(self.tfidf)

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

        self.clf = MultinomialNB()
        self.clf.fit(X, y)

        # validate
        if validate and len(validate) == 2:
            predictions = []

            for item in list(validate[0]):
                predictions.append(
                    self.clf.predict(item)
                )

            self.actual = validate[1]
            self.predicted = predictions

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

    def plot_cm(
        self,
        actual=None,
        predicted=None,
        filename='confusion_matrix.png'
    ):
        '''

        plot sentiment generated from 'vader_analysis'.

        '''

        if not actual:
            actual = self.actual
        if not predicted:
            predicted = self.predicted

        # generate plot
        plt.figure()
        skplt.metrics.plot_confusion_matrix(actual, predicted)

        # save plot
        plt.savefig(filename)
        plt.show()

    def get_accuracy(self, actual=None, predicted=None):
        '''

        return accuracy for prediction.

        '''

        if not actual:
            actual = self.actual
        if not predicted:
            predicted = self.predicted

        return(accuracy_score(actual, predicted))
