#!/usr/bin/python

import gensim


class LDA():
    '''

    apply LDA topic selection.

    '''

    def __init__(self, df, auto=True):
        '''

        define class variables.

        '''

        self.df = df

        if auto:
            self.train(self.df)

    def get_df(self):
        '''

        return current dataframe.

        '''

        return(self.df)

    def train(self, num_topics=10, passes=3, alpha='auto'):
        '''

        train lda model.

        @num_topics, number of topics to generate
        @passes, number of iterations for training.
        @alpha, defines dirichlet prior, where higher values correspond to more
            topics per document. When set to 'auto' parameter will tune itself.

        Note: this requires execution of 'self.normalize'.

        '''

        self.model = gensim.models.LdaModel(
            self.df,
            id2word=corpus.dictionary,
            alpha=alpha,
            num_topics=num_topics,
            passes=passes
        )

    def get_model(self):
        '''

        return trained model.

        '''

        return(self.model)

    def get_topics(self, num_topics=10):
        '''

        return most frequent topics.

        '''

        return(model.show_topic(num_topics))

    def get_topic_words(self, num_words=10):
        '''

        return most frequent words for most frequent topics.

        '''

        return([(
            i,
            [x for x, _ in self.model.get_topics(i, num_words)]
        ) for i in range(self.model.num_topics)])

    def predict(self, data):
        '''

        using trained model, predict topic for given data.

        Note: the number of topics returned depends on the train.

        '''

        data = data.split()
        doc_vector = self.model.id2word.doc2bow(doc)

        return(self.model[doc_vector])
