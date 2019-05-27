#!/usr/bin/python

import re
from pathlib import Path
from algorithm.text_classifier import Model as alg

def model(
    df=None,
    model_type='multinomial',
    key_text='text',
    key_class='screen_name',
    max_length=280,
    ngram=(1,1),
    split_size=0.2
):
    '''

    return trained classifier.

    '''

    # initialize classifier
    if df is not None:
        model = alg(
            df=df,
            key_text=key_text,
            key_class=key_class,
            ngram=ngram,
            split_size=split_size
        )
    else:
        model = alg(
            key_text=key_text,
            key_class=key_class,
            ngram=ngram,
            split_size=split_size
        )

    # vectorize data
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
    df,
    model_type='multinomial',
    key_text='SentimentText',
    key_class='Sentiment',
    max_length=280,
    stem=False,
    split_size=0.2
):
    '''

    return initialized model using pos.

    '''

    # initialize classifier
    if df is not None:
        model = alg(df=df, key_text=key_text, key_class=key_class, stem=False)
    else:
        model = alg(key_text=key_text, key_class=key_class, stem=False)

    #
    # suffix pos: add part of speech suffix to each word.
    #
    df['pos'] = [sent.split() for sent in df[key_text]]
    df['pos'] = [model.get_pos(x) for x in df['pos']]

    #
    # update model: using cleansed data.
    #
    model.set_df(df)
    model.set_key_text('pos')

    # vectorize data
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
