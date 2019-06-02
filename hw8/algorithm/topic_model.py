#!/usr/bin/python

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA


class Model():
    '''

    apply LDA topic selection.

    '''

    def __init__(self, df, auto=True):
        '''

        define class variables.

        '''

        self.df = df

        if auto:
            self.vectorize()
            self.train(self.df)

    def get_df(self):
        '''

        return current dataframe.

        '''

        return(self.df)

    def vectorize(
        self,
        max_df=0.95,
        min_df=0.2,
        max_features=30,
        stop_words='english',
        model_type=None
    ):
        '''

        vectorize provided data.

        @max_df, ignore terms that have a document frequency strictly higher
            than the given threshold.
        @min_df, ignore terms that have a document frequency strictly lower
            than the given threshold.
        @max_features, only consider the top max_features ordered by term
            frequency across the corpus.

        Note: 'max_df', and 'min_df' float parameter represents a proportion
            of documents, while integer absolute counts.

        '''

        # term frequency-inverse document frequency
        if model_type == 'nmf':
            self.vectorizer = TfidfVectorizer(
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                stop_words=stop_words
            )

        # term frequency
        else:
            self.vectorizer = CountVectorizer(
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                stop_words=stop_words
            )

        self.fit = self.vectorizer.fit_transform(self.df)

    def get_fit(self):
        '''

        get current fitted vectorizer.

        '''

        return(self.fit)

    def get_feature_names(self):
        '''

        get current feature names.

        '''

        return(self.vectorizer.get_feature_names())

    def train(
        self,
        model_type=None,
        num_components=20,
        random_state=1,
        alpha=.1,
        l1_ratio=.5,
        init='nndsvd',
        num_topics=40,
        max_iter=5,
        learning_method='online',
        learning_offset=50.
    ):
        '''

        train topic model.

        NMF:
        @n_components, number of features (defaults to all)
        @alpha, regularization items
        @l1_ratio, regularization mixing parameters
        @init, used to initialize the procedure
            - nndsvd, better for sparseness

        LDA:
        @n_components, number of topics
        @max_iter, maximum number of iterations
        @learning_method, method used to update _component
            - online is much faster for bigger data
        @learning_offset, parameter that downweights early iterations in
            online learning

        '''

        if model_type == 'nmf':
            self.model = NMF(
                n_components=num_components,
                random_state=random_state,
                alpha=alpha,
                l1_ratio=l1_ratio,
                init=init
            ).fit(self.fit)

        else:
            self.model = LDA(
                n_topics=num_topics,
                max_iter=max_iter,
                learning_method=learning_method,
                learning_offset=learning_offset,
                random_state=random_state
            ).fit(self.fit)

    def get_model(self):
        '''

        return trained model.

        '''

        return(self.model)

    def get_topic_words(self, feature_names, num_words=10):
        '''

        return most frequent words for most frequent topics.

        '''

        return([(topic_idx,
            [feature_names[i]
                for i in topic.argsort()[:-num_words - 1:-1]])
                    for topic_idx,
                        topic in enumerate(self.model.components_)])

    def predict(self, data):
        '''

        using trained model, predict topic for given data.

        Note: the number of topics returned depends on the train.

        '''

        data = data.split()
        doc_vector = self.model.id2word.doc2bow(doc)

        return(self.model[doc_vector])
