#!/usr/bin/python

from model.lda import model
from view.classifier import plot_bar

def lda(
    num_topics=10,
    num_words=10,
    passes=3,
    alpha='auto',
    auto=True,
    rotation=90,
    directory='viz',
    flag_lda=True,
    plot=True
):
    '''

    implement topic model.

    '''

    if flag_lda:
        lda = model(
            df=df,
            auto=auto,
        )

        if not auto:
            lda.train(
                num_topics=10,
                passes=3,
                alpha=alpha
            )

        topics = self.model.get_topics(num_topics)
        topic_words = self.model.get_topic_words(num_words)

        if plot:
            plot_bar(
                labels=[i[0] for i in topics],
                performance=[i[1] for i in topics],
                directory=directory,
                filename='lda_topics-{num_topics}'.format(
                    num_topics=num_topics
                ),
                rotation=rotation
            )

            plot_bar(
                labels=[i[0] for i in topic_words],
                performance=[i[1] for i in topic_words],
                directory=directory,
                filename='lda_words-{num_words}'.format(
                    num_words=num_words
                ),
                rotation=rotation
            )
