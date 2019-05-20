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
from scipy.stats import itemfreq
from nltk.corpus import stopwords
from nltk import tokenize, download, pos_tag
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import scikitplot as skplt
from algorithm.penn_treebank import penn_scale
from utility.dataframe import cleanse
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
        stem=True,
        lowercase=True,
        cleanse_data=True,
        fp='{}/data/sample-sentiment.csv'.format(
            Path(__file__).resolve().parents[1]
        )
    ):
        '''

        define class variables.

        '''

        # class variables
        self.key_text = key_text
        self.key_class = key_class
        self.actual = None
        self.predicted = None

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(fp)

        # clean text
        if cleanse_data:
            self.df[self.key_text] = cleanse(self.df, self.key_text)

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

    def set_df(self, df):
        '''

        update dataframe.

        '''

        self.df = df

    def set_key_text(self, key_text):
        '''

        update key text.

        '''

        self.key_text = key_text

    def split(self, size=0.20):
        '''

        split data into train and test.

        '''

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.tfidf,
            self.df[self.key_class],
            test_size=size
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

    def get_feature_distribution(self):
        '''

        get feature distribution on given dataset.

        '''

        return({
            'y_train': itemfreq(self.y_train),
            'y_test': itemfreq(self.y_test),
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

    def vectorize(
        self,
        data=None,
        stop_words='english',
        ngram_range=(1,1),
        topn=25
    ):
        '''

        vectorize provided data.

        '''

        #
        # pos case: implemented internally by 'split' when 'pos_split=True'.
        #
        if data is not None:
            # bag of words: with 'english' stopwords
            count_vect = CountVectorizer(
                stop_words=stop_words,
                ngram=ngram
            )
            bow = count_vect.fit_transform(data)

            # tfidf weighting
            tfidf_vectorizer = TfidfVectorizer(
                stop_words=stop_words,
                ngram=ngram
            )
            tfidf = tfidf_vectorizer.fit_transform(data)

            return(bow, tfidf)

        #
        # general case
        #
        else:
            # bag of words: with 'english' stopwords
            self.count_vect = CountVectorizer(
                stop_words=stop_words,
                ngram=ngram
            )
            self.bow = self.count_vect.fit_transform(self.df[self.key_text])

            # tfidf weighting
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words=stop_words,
                ngram=ngram
            )
            self.tfidf = self.tfidf_vectorizer.fit_transform(self.df[self.key_text])

            # top n tfidf words
            feature_names = self.count_vect.get_feature_names()
            sorted_items = self.sort_coo(self.tfidf.tocoo())
            self.keywords = self.get_top_features(
                feature_names,
                sorted_items,
                topn
            )

    def sort_coo(self, coo_matrix):
        '''

        return sorted vector values while preserving the column index.

        '''

        tuples = zip(coo_matrix.col, coo_matrix.data)
        return(sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True))

    def get_top_features(self, feature_names, sorted_items, topn):
        '''

        return feature names with associated tf-idf score of top n words.

        '''

        # use only topn items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:

            # keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        # create a tuples of feature,score
        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]

        return(results)

    def get_feature_names(self):
        '''

        get feature names for current dataframe.

        '''

        return(self.count_vect.get_feature_names())

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

    def train(
        self,
        X,
        y,
        validate=False,
        max_length=280,
        model_type=None,
        multiclass=False
    ):
        '''

        train classifier model.

        @validate, must have tuple shape (X_test, y_test)
        @model_type, override default behavior with defined model.

            - bernoulli
            - multinomial (default)
            - svm, with linear kernel since text is high dimensional.

        @multiclass, svm indicator of greater than binary classification.

        '''

        # conditionally select model
        if (model_type == 'svm'):
            if multiclass:
                clf = svm.SVC(gamma='scale', kernel='linear', decision_function_shape='ovo')
            else:
                clf = svm.SVC(gamma='scale', kernel='linear')

            clf.fit(X, y)

            # validate
            if validate and len(validate) == 2:
                predictions = []

                for item in list(validate[0]):
                    predictions.append(
                        clf.predict(item)
                    )

                self.actual = validate[1]
                self.predicted = predictions

                return({
                    'model': clf,
                    'actual': validate[1],
                    'predicted': predictions
                })

            return({
                'model': clf,
                'actual': None,
                'predicted': None
            })

        elif (
            (model_type == 'bernoulli') or
            (not model_type and all(len(sent) <= max_length for sent in self.X_train))
        ):
            clf = BernoulliNB()
            clf.fit(X, y)

            # validate
            if validate and len(validate) == 2:
                predictions = []

                for item in list(validate[0]):
                    predictions.append(
                        clf.predict(item)
                    )

                self.actual = validate[1]
                self.predicted = predictions

                return({
                    'model': clf,
                    'actual': validate[1],
                    'predicted': predictions
                })

            return({
                'model': clf,
                'actual': None,
                'predicted': None
            })

        else:
            clf = MultinomialNB()
            clf.fit(X, y)

            # validate
            if validate and len(validate) == 2:
                predictions = []

                for item in list(validate[0]):
                    predictions.append(
                        clf.predict(item)
                    )

                self.actual = validate[1]
                self.predicted = predictions

                return({
                    'model': clf,
                    'actual': validate[1],
                    'predicted': predictions
                })

            return({
                'model': clf,
                'actual': None,
                'predicted': None
            })

    def plot_cm(
        self,
        actual=None,
        predicted=None,
        filename='confusion_matrix.png',
        show=False
        rotation=90
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
        skplt.metrics.plot_confusion_matrix(
            actual,
            predicted,
            x_tick_rotation=rotation
        )

        # save plot
        plt.savefig(filename)

        if show:
            plt.show()
        else:
            plt.close()

    def get_accuracy(self, actual=None, predicted=None):
        '''

        return accuracy for prediction.

        '''

        if not actual:
            actual = self.actual
        if not predicted:
            predicted = self.predicted

        return(accuracy_score(actual, predicted))


    def get_kfold_scores(
        self,
        n_splits=2,
        stop_words='english',
        max_length=280,
        shuffle=True,
        model_type=None,
        multiclass=False,
        ngram=(1,1)
    ):
        '''

        return kfold validation scores. Variance between scores is
        an indication that either the algorithm is unable to learn,
        or the data may require additional cleaning and preprocessing.

        @model_type, override default behavior with defined model.

            - bernoulli
            - multinomial (default)
            - svm, with linear kernel since text is high dimensional.

        @multiclass, svm indicator of greater than binary classification.

        '''

        # bag of words: with 'english' stopwords
        count_vect = CountVectorizer(
            stop_words=stop_words,
            ngram=ngram
        )
        bow = self.count_vect.fit_transform(self.df[self.key_text])

        # conditionally select model
        if (model_type == 'svm'):
            if multiclass:
                clf = svm.SVC(
                    gamma='scale',
                    kernel='linear',
                    decision_function_shape='ovo'
                )
            else:
                clf = svm.SVC(gamma='scale', kernel='linear')

            # tfidf weighting
            tfidf_vectorizer = TfidfVectorizer(
                stop_words=stop_words,
                ngram=ngram
            )
            data = tfidf_vectorizer.fit_transform(self.df[self.key_text])

        elif (
            (model_type == 'bernoulli') or
            (not model_type and all(len(sent) <= max_length for sent in self.X_train))
        ):
            clf = BernoulliNB()
            data = bow

        else:
            clf = MultinomialNB()
            tfidf_vectorizer = TfidfVectorizer(
                stop_words=stop_words,
                ngram=ngram
            )
            data = tfidf_vectorizer.fit_transform(self.df[self.key_text])

        # random kfolds
        return(
            cross_val_score(
                clf,
                data,
                y=self.df[self.key_class],
                cv=KFold(n_splits, shuffle=True)
            )
        )
