#!/usr/bin/python

#
# this files requires nltk to be installed:
#
#     pip install nltk
#     pip install -U scikit-learn
#     pip install scikit-plot
#

import os
import time
import csv
import numpy as np
from pathlib import Path
import pandas as pd
from nltk import tokenize, download
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import scikitplot as skplt
download('vader_lexicon')


def vader_analysis(fp='{}/data/sample-sentiment.csv'.format(
    Path(__file__).resolve().parents[1]
)):
    '''

    use vader to perform sentiment analyis.

    '''

    sentences = pd.read_csv(fp)['SentimentText']
    sid = SentimentIntensityAnalyzer()

    result = []
    for i, s in enumerate(sentences):
        ss = sid.polarity_scores(s)

        scores = []
        for k in sorted(ss):
            scores.append({k: ss[k]})

        result.append({i: scores})

    return({'sent': sentences, 'result': result})

def nb_model(fp='{}/data/sample-sentiment.csv'.format(
    Path(__file__).resolve().parents[1]
)):
    '''

    use sklearn and nltk packages to create naive bayes model.

    '''

    data = pd.read_csv(fp)
    X_train, X_test, y_train, y_test = train_test_split(
        data['SentimentText'],
        data['Sentiment'],
        test_size=0.25
    )

    # bag of words: with 'english' stopwords
    count_vect = CountVectorizer(stop_words='english')
    bow = count_vect.fit_transform(X_train)

    # tfidf weighting
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(bow)

    # fit model
    clf = MultinomialNB().fit(X_train_tfidf, y_train)

    # predict
    predictions = []
    for item in list(X_test):
        prediction = count_vect.transform([item])
        predictions.append(
            clf.predict(tfidf_transformer.fit_transform(prediction))
        )

    # fit model
    return({
        'model': clf,
        'actual': y_test,
        'predicted': predictions
    })

def time_df(fp='{}/data/sample-sentiment.csv'.format(
    Path(__file__).resolve().parents[1]
)):
    '''

    compare upload time for provided csv.

    '''

    # panda
    start_pd = time.time()
    df_pd = pd.read_csv(fp)
    pd_time = time.time() - start_pd

    # numpy + panda
    start_np = time.time()
    with open(fp, 'r') as f:
        data = list(csv.reader(f, delimiter=','))
        df_np = pd.DataFrame(data[1:])
        df_np.columns = data[0]
    np_time = time.time() - start_np

    return({
        'pd_time': pd_time,
        'np_time': np_time,
        'pd_size': df_pd.size,
        'np_size': df_np.size
    })

if __name__ == '__main__':
    # create viz directory
    if not os.path.exists('viz'):
        os.makedirs('viz')

    # dataframe benchmark
    tdf = time_df()
    print('panda upload time: {}'.format(tdf['pd_time']))
    print('numpy upload time: {}'.format(tdf['np_time']))

    objects = ('panda', 'numpy')
    y_pos = np.arange(len(objects))
    performance = [tdf['pd_time'], tdf['np_time']]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Performance')
    plt.savefig('viz/sentiment')
    plt.show()

    # vader analysis
    va = vader_analysis()
    [print('{}\n{}\n\n'.format(x, va['result'][i])) for i,x in enumerate(va['sent'])]

    # naive bayes prediction
    model = nb_model()
    skplt.metrics.plot_confusion_matrix(
        model['actual'],
        model['predicted']
    )
    plt.savefig('viz/confusion_matrix')
    plt.show()
