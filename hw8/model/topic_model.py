#!/usr/bin/python

import re
from pathlib import Path
from algorithm.topic_model import Model

def model(
    model_type=None,
    n_components=20,
    random_state=1,
    alpha=.1,
    l1_ratio=.5,
    init='nndsvd',
    num_topics=40,
    max_iter=5,
    learning_method='online',
    learning_offset=50.,
    auto=False
):
    '''

    return lda categorization.

    '''

    if not auto:
        model = Model(auto=False)

    else:
        lda = Model()

    model.vectorize(
        max_df=0.95,
        min_df=2,
        stop_words='english',
        model_type=None
    )

    if model_type == 'nmf':
        model.train(
            n_components=n_components,
            random_state=random_state,
            alpha=alpha,
            l1_ratio=l1_ratio,
            init=init
        )

    else:
        model.train(
            n_topics=num_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            learning_offset=learning_offset,
            random_state=random_state
        )

    return(model)
