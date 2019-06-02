#!/usr/bin/python

import re
from pathlib import Path
from algorithm.topic_model import Model

def model(
    df,
    max_df=0.95,
    min_df=0.2,
    model_type=None,
    num_components=20,
    random_state=1,
    alpha=.1,
    l1_ratio=.5,
    init='nndsvd',
    num_topics=40,
    max_iter=5,
    learning_method='online',
    learning_offset=50.,
    vectorize_stopwords='english',
    auto=False
):
    '''

    return lda categorization.

    '''

    if not auto:
        model = Model(df=df, auto=False)
        model.vectorize(
            max_df=max_df,
            min_df=min_df,
            model_type=None,
            stop_words=vectorize_stopwords
        )

    else:
        lda = Model(df=df)

    if model_type == 'nmf':
        model.train(
            num_components=num_components,
            random_state=random_state,
            alpha=alpha,
            l1_ratio=l1_ratio,
            init=init
        )

    else:
        model.train(
            num_topics=num_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            learning_offset=learning_offset,
            random_state=random_state
        )

    return(model)
