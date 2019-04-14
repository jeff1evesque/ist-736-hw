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
from nltk.corpus import stopwords
from nltk import tokenize, download, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import scikitplot as skplt
from penn_treebank import penn_scale
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

    def split(self, test_size=0.20, stem=True):
        '''

        split data into train and test.

        '''

        # clean
        self.df[self.key_text] = [x.strip('#') for x in self.df[self.key_text]]

        # stem
        if stem:
            porter = PorterStemmer()
            self.df[self.key_text] = [porter.stem(word) for word in self.df[self.key_text]]

        # split
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

    def get_pos(self, l, pos_length=40):
        '''

        apply pos tagger to supplied list.

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
    df_pos['pos'] = unigram.get_pos(
        df_pos['SentimentText'].apply(lambda x: x.split())
    )[1]
    print(df_pos)
