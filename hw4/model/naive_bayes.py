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
import numpy as np
from pathlib import Path
import pandas as pd
from nltk.corpus import stopwords
from nltk import tokenize, download, pos_tag
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
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

        # vectorize data
        if vectorize:
            self.split()
            self.vectorize()

    def split(self, test_size=0.20, stem=True, pos_split=False):
        '''

        split data into train and test.

        '''

        # clean
        self.df[self.key_text] = [re.sub('[#@]', '', x) for x in self.df[self.key_text]]

        # stem
        if stem:
            porter = PorterStemmer()
            self.df[self.key_text] = [porter.stem(word) for word in self.df[self.key_text]]

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

    def get_pos(self, l, pos_length=280):
        '''

        apply pos tagger to supplied list.

        @pos_length, maximum number of words in tweets.

        '''

        result_word = []
        result_pos = []
        pos = [pos_tag(x) for x in l]
        for y in pos:
            result_word.append([x[0] for x in y if x[0] not in stop_words])
            result_pos.append(
                [penn_scale[x[1]] if x[1] in penn_scale and x[0] not in stop_words else 1 for x in y]
            )

        # consistent length
        for i,x in enumerate(result_pos):
            if len(x) < pos_length:
                difference = pos_length - len(x)
                result_pos[i].extend([1] * difference)
            else:
                difference = len(x) - pos_length
                result_pos[i] = result_pos[i][:len(result_pos[i]) - difference]

        return(result_word, result_pos)

    def vectorize(self, stop_words='english'):
        '''

        vectorize provided data.

        '''

        # bag of words: with 'english' stopwords
        self.count_vect = CountVectorizer(stop_words=stop_words)
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

    def model(self, X, y, validate=False, max_length=280):
        '''

        create naive bayes model.

        @validate, must have tuple shape (X_test, y_test)

        '''

        # conditionally select model
        if all(len(sent) <= max_length for sent in self.X_train):
            self.clf = BernoulliNB()
        else:
            self.clf = MultinomialNB()

        # fit model
        self.fit(X, y)

        # validate
        if validate and len(validate) == 2:
            predictions = []
            for item in list(validate[0]):
                prediction = self.count_vect.transform([item])
                predictions.append(
                    self.clf.predict(self.tfidf_transformer.fit_transform(prediction))
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
