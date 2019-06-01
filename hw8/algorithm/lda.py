#!/usr/bin/python

import gensim


class LDA():
    '''

    apply LDA topic selection.

    '''

    def __init__(self, df):
        '''

        define class variables.

        @num_topics, number of topics to generate
        @passes, number of iterations for training.
        @alpha, defines dirichlet prior, where higher values correspond to more
            topics per document. When set to 'auto' parameter will tune itself.

        '''

        self.df = df
        self.train(df)

    def get_df(self):
        '''

        return current dataframe.

        '''

        return(self.df)

    def train(self, num_topics=10, passes=3, alpha='auto'):
        '''

        split data into train and test.

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

    def get_topic(self, num_topics=10):
        '''

        return most frequent topics.

        '''

        return(model.show_topic(num_topics))

    def get_topic_words(self, num_topics, num_words):
        '''

        return most frequent words for most frequent topics.

        '''

        return([(
            topic_id,
            [x for x, _ in self.model.get_topic(topic_id, num_words)]
        ) for topic_id in range(self.model.num_topics)])

    def predict(self, data):
        '''

        using trained model, predict topic for given data.

        Note: the number of topics returned depends on the train.

        '''

        data = data.split()
        doc_vector = self.model.id2word.doc2bow(doc)

        return(self.model[doc_vector])
