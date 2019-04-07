#!/usr/bin/python

#
# this files requires nltk to be installed:
#
#     pip install nltk
#     pip install -U scikit-learn
#

import time
import csv
import numpy as np
from pathlib import Path
import pandas as pd
from nltk import tokenize, download
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
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

def custom_analysis(fp='{}/data/sample-sentiment.csv'.format(
    Path(__file__).resolve().parents[1]
)):
    '''

    use sklearn and nltk packages to perform classification analysis.

    '''

    data = pd.read_csv(fp)
    X_train, X_test, y_train, y_test = train_test_split(
        data['SentimentText'],
        data['Sentiment'],
        test_size=0.25
    )

    # bag of words: with 'english' stopwords
    count_vect = CountVectorizer(stop_words='english')
    bow = count_vect(train_text)

    # tfidf weighting
    tfidf = TfidfVectorizer()
    tfidf = transformer.fit_transform(bow)

    # fit model
    clf = MultinomialNB().fit(tfidf, y_train)

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
    tdf = time_df()
    print('panda upload time: {}'.format(tdf['pd_time']))
    print('numpy upload time: {}'.format(tdf['np_time']))

    va = vader_analysis()
    [print('{}\n{}\n\n'.format(x, va['result'][i])) for i,x in enumerate(va['sent'])]
