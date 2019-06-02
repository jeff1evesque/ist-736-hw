#!/usr/bin/python

from model.topic_model import model
from view.plot import plot_bar

def topic_model(
    df,
    n_components=20,
    random_state=1,
    alpha=.1,
    l1_ratio=.5,
    init='nndsvd',
    num_topics=40,
    max_iter=5,
    learning_method='online',
    learning_offset=50.,
    auto=False,
    directory='viz',
    rotation=90,
    flag_lda=True,
    flag_nmf=True,
    plot=True
):
    '''

    implement topic model.

    '''

    if flag_lda:
        lda = model(
            df=df,
            n_topics=num_topics,
            max_iter=max_iter,
            learning_method=learning_method,
            learning_offset=learning_offset,
            random_state=random_state
        )
        topics = lda.get_topics(num_topics=num_topics)
        topic_words = lda.get_topic_words(num_words=num_words)

        if plot:
            plot_bar(
                labels=[i[0] for i in topics[0]],
                performance=[i[1] for i in topics[0]],
                directory=directory,
                filename='lda_topics-{num_topics}'.format(
                    num_topics=num_topics
                ),
                rotation=rotation
            )

            plot_bar(
                labels=[i[0] for i in topic_words[1]],
                performance=[i[1] for i in topic_words[1]],
                directory=directory,
                filename='lda_words-{num_words}'.format(
                    num_words=num_words
                ),
                rotation=rotation
            )

    if flag_nmf:
        nmf = model(
            df=df,
            n_components=n_components,
            random_state=random_state,
            alpha=alpha,
            l1_ratio=l1_ratio,
            init=init
        )
        topics = nmf.get_topics(num_topics=num_topics)
        topic_words = nmf.get_topic_words(num_words=num_words)

        if plot:
            plot_bar(
                labels=[i[0] for i in topics[0]],
                performance=[i[1] for i in topics[0]],
                directory=directory,
                filename='nmf_topics-{num_topics}'.format(
                    num_topics=num_topics
                ),
                rotation=rotation
            )

            plot_bar(
                labels=[i[0] for i in topic_words[1]],
                performance=[i[1] for i in topic_words[1]],
                directory=directory,
                filename='nmf_words-{num_words}'.format(
                    num_words=num_words
                ),
                rotation=rotation
            )
