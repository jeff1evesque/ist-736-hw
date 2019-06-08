#!/usr/bin/python

import re
from pathlib import Path
from algorithm.topic_model import Model as alg

def model_lda(
    df,
    key_text='text',
    max_df=1.0,
    min_df=1,
    random_state=0,
    num_topics=10,
    max_features=500,
    max_iter=5,
    learning_method='online',
    learning_offset=50.,
    vectorize_stopwords='english',
    stopwords=[],
    auto=False,
    ngram=1
):
    '''

    return topic model categorization.

    '''

    model = alg(
        df[key_text],
        auto=auto,
        stopwords=stopwords,
        ngram=ngram
    )

    if not auto:
        model.vectorize(
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            model_type='lda',
            stopwords=vectorize_stopwords
        )

    model.train(
        num_topics=num_topics,
        max_iter=max_iter,
        learning_method=learning_method,
        learning_offset=learning_offset,
        random_state=random_state
    )

    return(model)

def model_nmf(
    df,
    key_text='text',
    max_df=1.0,
    min_df=1,
    random_state=None,
    alpha=.1,
    l1_ratio=.5,
    init='nndsvd',
    num_topics=10,
    max_features=500,
    vectorize_stopwords='english',
    stopwords=[],
    auto=False,
    ngram=1
):
    '''

    return topic model categorization.

    '''

    model = alg(
        df[key_text],
        auto=auto,
        stopwords=stopwords,
        ngram=ngram
    )

    if not auto:
        model.vectorize(
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            model_type='nmf',
            stopwords=vectorize_stopwords
        )

    model.train(
        num_topics=num_topics,
        random_state=random_state,
        alpha=alpha,
        l1_ratio=l1_ratio,
        init=init
    )

    return(model)
