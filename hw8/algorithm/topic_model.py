#!/usr/bin/python

import numpy as np
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from nltk.corpus import stopwords as stp
from utility.dataframe import cleanse
stop_english = stp.words('english')

class Model():
    '''

    apply topic modeling.

    '''

    def __init__(
        self,
        df,
        key_text='text',
        auto=True,
        stopwords=[],
        ngram=1,
        lowercase=True,
        cleanse_data=True
    ):
        '''

        define class variables.

        '''

        self.df = df
        self.key_text = key_text
        stopwords.extend(stop_english)
        self.stopwords = stopwords

        # clean text
        if cleanse_data:
            self.df[self.key_text] = cleanse(self.df, self.key_text)

        if lowercase:
            self.df[self.key_text] = [w.lower()
                for w in self.df[self.key_text]]

        self.df[self.key_text] = self.df[self.key_text].apply(
            lambda x: [' '.join(
                [w for w in x.split(' ') if w.strip() not in self.stopwords]
            )][0]
        )

        if ngram > 1:
            self.df[self.key_text] = [self.create_ngram(s, n=ngram)
                for s in self.df[self.key_text]]
            self.df[self.key_text] = [' '.join(x)
                for x in self.df[self.key_text]]

        if auto:
            self.vectorize()
            self.train(self.df)

    def get_df(self):
        '''

        return current dataframe.

        '''

        return(self.df)

    def create_ngram(self, sentence, n=2):
        '''

        generate ngram on supplied sentence.

        '''

        tokens = [token for token in sentence.split(' ') if token != '']
        ng = zip(*[tokens[i:] for i in range(n)])
        return(['_'.join(ngram) for ngram in ng])

    def vectorize(
        self,
        max_df=1.0,
        min_df=1,
        max_features=None,
        stopwords='english',
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
                stop_words=stopwords
            )

        # term frequency
        else:
            self.vectorizer = CountVectorizer(
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                stop_words=stopwords
            )

        self.fit = self.vectorizer.fit_transform(self.df[self.key_text])

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
        random_state=1,
        alpha=.1,
        l1_ratio=.5,
        init='nndsvd',
        num_topics=20,
        max_iter=5,
        learning_method='online',
        learning_offset=50.
    ):
        '''

        train topic model.

        NMF:
        @n_components, number of topics (defaults to all)
        @alpha, regularization items
        @l1_ratio, regularization mixing parameters
        @init, used to initialize the procedure
            - nndsvd, better for sparseness

        LDA:
        @n_components, number of topics (defaults to all)
        @max_iter, maximum number of iterations
        @learning_method, method used to update _component
            - online is much faster for bigger data
        @learning_offset, parameter that downweights early iterations in
            online learning

        '''

        if model_type == 'nmf':
            self.model = NMF(
                n_components=num_topics,
                random_state=random_state,
                alpha=alpha,
                l1_ratio=l1_ratio,
                init=init
            ).fit(self.fit)

        else:
            self.model = LDA(
                n_components=num_topics,
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

    def get_topic_words(self, feature_names, num_words=20):
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
