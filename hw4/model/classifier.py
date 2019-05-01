#!/usr/bin/python

import re
from pathlib import Path
from algorithm.text_classifier import Model as alg

def model(
    df=None,
    model_type='multinomial',
    key_text='text',
    key_class='screen_name',
    max_length=280
):
    '''

    return trained classifier.

    '''

    # initialize classifier
    if df is not None:
        model = alg(df=df, key_text=key_text, key_class=key_class)
    else:
        model = alg(key_text=key_text, key_class=key_class)

    # vectorize data
    vectorized = model.get_tfidf()
    model.split()
    params = model.get_split()

    # train classifier
    model.train(
        params['X_train'],
        params['y_train'],
        model_type=model_type,
        validate=(params['X_test'], params['y_test']),
        max_length=max_length
    )

    return(model)

def model_pos(
    m,
    model_type='multinomial',
    key_text='SentimentText',
    key_class='Sentiment',
    max_length=280
):
    '''

    return initialized model using pos.

    '''

    # reduce to ascii
    regex = r'[^\x00-\x7f]'
    df_m = m.get_df()
    df_m['pos'] = [re.sub(regex, r' ', sent).split() for sent in df_m[key_text]]
	
    # suffix pos
    df_m['pos'] = [m.get_pos(x) for x in df_m['pos']]
    model = alg(df=df_m, key_class=key_class, key_text=key_text)

    # vectorize data
    vectorized = model.get_tfidf()
    model.split()
    params = model.get_split()

    # train classifier
    model.train(
        params['X_train'],
        params['y_train'],
        model_type=model_type,
        validate=(params['X_test'], params['y_test']),
        max_length=max_length
    )

    return(model)
