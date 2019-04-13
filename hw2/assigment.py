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
from nltk import tokenize, download
from nltk.tag import StanfordPOSTagger
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

        # instantiate
        self.split()
        self.vectorize()

        # pos dataframe
        self.pos = self.data
        self.pos[self.key_value] = self.get_pos(self.data[self.key_value])

    def split(self, test_size=0.25):
        '''

        split data into train and test.

        '''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data[key_text],
            self.data[key_class],
            test_size=test_size
        )

    def get_pos(self, l, tagger='english-bidirectional-distsim.tagger'):
        '''

        apply stanford pos tagger to supplied list.

        '''

        st = StanfordPOSTagger(tagger)
        return([st.tag(x) for x in l])

    def vectorize(self):
        '''

        vectorize provided data.

        '''

        # bag of words: with 'english' stopwords
        count_vect = CountVectorizer(stop_words='english')
        bow = count_vect.fit_transform(self.X_train)

        # tfidf weighting
        tfidf_transformer = TfidfTransformer()
        self.X_train_tfidf = tfidf_transformer.fit_transform(bow)

    def nb_model(self):
        '''

        use sklearn and nltk packages to create naive bayes model.

        '''

        # fit model
        self.clf = MultinomialNB().fit(self.X_train_tfidf, self.y_train)

        # predict
        predictions = []
        for item in list(self.X_test):
            prediction = count_vect.transform([item])
            predictions.append(
                self.clf.predict(tfidf_transformer.fit_transform(prediction))
            )

        # fit model
        return({
            'model': clf,
            'actual': self.y_test,
            'predicted': predictions
        })

if __name__ == '__main__':
    # naive bayes prediction
    model = nb_model()
    skplt.metrics.plot_confusion_matrix(
        model['actual'],
        model['predicted']
    )
    plt.show()
