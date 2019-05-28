#!/usr/bin/python

import math
from collections import OrderedDict
from model.classifier import model as m_model
from model.classifier import model_pos as mp_model
from view.classifier import plot_cm, plot_bar

def classify(
    df,
    key_class='screen_name',
    key_text='full_text',
    kfold=True,
    prf=True,
    n_splits=5,
    top_words=20,
    ngram=(1,1),
    rotation=90,
    directory='viz',
    flag_mnb=True,
    flag_mnb_pos=True,
    flag_bnb=True,
    flag_bnb_pos=True,
    flag_svm=True,
    flag_svm_pos=True,
    plot=True,
    split_size=0.2,
    validate=True,
    stopwords=None
):
    '''

    implement designated classifiers.

    '''

    # local variables
    kfold_scores = {}
    model_scores = {}
    prf_scores = {}
    indicative_words = {}

    if ngram == (1,1):
        suffix = ''
    else:
        suffix = '_ngram'

    # multinomial naive bayes
    if flag_mnb:
        mnb = m_model(
            df=df,
            key_class=key_class,
            key_text=key_text,
            max_length=math.inf,
            ngram=ngram,
            split_size=split_size,
            validate=validate,
            stopwords=stopwords
        )
        model_scores['mnb'] = mnb.get_accuracy()

        # most indicative words
        log_prob = mnb.get_word_scores(
            mnb.get_clf(),
            top_words=top_words
        )

        terms = mnb.get_count_vect().get_feature_names()
        indicative_words['positive'] = [(
            log_prob['positive']['value'][i],
            terms[x]
        ) for i,x in enumerate(log_prob['positive']['index'])]
        indicative_words['positive'] = sorted(
            indicative_words['positive']
        )

        indicative_words['negative'] = [(
            log_prob['negative']['value'][i],
            terms[x]
        ) for i,x in enumerate(log_prob['negative']['index'])]
        indicative_words['negative'] = sorted(
            indicative_words['negative']
        )

        if plot:
            plot_cm(
                mnb,
                directory=directory,
                file_suffix='{key_class}{suffix}'.format(
                    key_class=key_class,
                    suffix=suffix
                )
            )

            # extract top n words
            tfidf = mnb.get_tfidf()
            keywords = mnb.get_top_features(
                mnb.get_feature_names(),
                mnb.sort_coo(tfidf.tocoo()),
                top_words
            )

            # sort values: largest to smallest
            keywords = OrderedDict(
                sorted(keywords.items(),
                key=lambda x: x[0])
            )

            # plot top n words
            plot_bar(
                labels=[*keywords],
                performance=[*keywords.values()],
                directory=directory,
                filename='top_{count}_tfidf{suffix}'.format(
                    count=top_words,
                    suffix=suffix
                ),
                rotation=rotation
            )

            # feature distribution
            train = mnb.get_feature_distribution()['y_train']
            test = mnb.get_feature_distribution()['y_test']
            plot_bar(
                labels=[x[0] for x in train],
                performance=[x[1] for x in train],
                directory=directory,
                filename='train_distribution_mnb{suffix}'.format(
                    suffix=suffix
                ),
                rotation=rotation
            )
            plot_bar(
                labels=[x[0] for x in test],
                performance=[x[1] for x in test],
                directory=directory,
                filename='test_distribution_mnb{suffix}'.format(
                    suffix=suffix
                ),
                rotation=rotation
            )

            # plot top n words
            plot_bar(
                labels=[x[1] for x in indicative_words['positive']],
                performance=[x[0] for x in indicative_words['positive']],
                directory=directory,
                filename='top_{count}_positive_words_mnb'.format(
                    count=top_words
                ),
                rotation=rotation
            )

            plot_bar(
                labels=[x[1] for x in indicative_words['negative']],
                performance=[x[0] for x in indicative_words['negative']],
                directory=directory,
                filename='top_{count}_negative_words_mnb'.format(
                    count=top_words
                ),
                rotation=rotation
            )

        if kfold:
            kfold_scores['mnb'] = mnb.get_kfold_scores(
                model_type='multinomial',
                n_splits=n_splits,
                ngram=ngram
            )

        if prf:
            prf_scores['mnb'] = mnb.get_precision_recall_fscore()

    if flag_mnb_pos:
        mnb_pos = mp_model(
            df=df,
            key_class=key_class,
            key_text=key_text,
            max_length=math.inf,
            split_size=split_size,
            validate=validate,
            stopwords=stopwords
        )
        model_scores['mnb_pos'] = mnb_pos.get_accuracy()

        # most indicative words
        log_prob = mnb_pos.get_word_scores(
            mnb_pos.get_clf(),
            top_words=top_words
        )

        terms = mnb.get_count_vect().get_feature_names()
        indicative_words['positive'] = [(
            log_prob['positive']['value'][i],
            terms[x]
        ) for i,x in enumerate(log_prob['positive']['index'])]
        indicative_words['positive'] = sorted(
            indicative_words['positive']
        )

        indicative_words['negative'] = [(
            log_prob['negative']['value'][i],
            terms[x]
        ) for i,x in enumerate(log_prob['negative']['index'])]
        indicative_words['negative'] = sorted(
            indicative_words['negative']
        )

        if plot:
            plot_cm(
                mnb_pos,
                directory=directory,
                file_suffix='{}_pos'.format(key_class)
            )

            # extract top n words
            tfidf = mnb_pos.get_tfidf()
            keywords = mnb_pos.get_top_features(
                mnb_pos.get_feature_names(),
                mnb_pos.sort_coo(tfidf.tocoo()),
                top_words
            )

            # sort values: largest to smallest
            keywords = OrderedDict(
                sorted(keywords.items(),
                key=lambda x: x[0])
            )

            # plot top n words
            plot_bar(
                labels=[*keywords],
                performance=[*keywords.values()],
                directory=directory,
                filename='top_{count}_tfidf'.format(count=top_words),
                rotation=rotation
            )

            # feature distribution
            train = mnb_pos.get_feature_distribution()['y_train']
            test = mnb_pos.get_feature_distribution()['y_test']
            plot_bar(
                labels=[x[0] for x in train],
                performance=[x[1] for x in train],
                directory=directory,
                filename='train_distribution_mnb_pos',
                rotation=rotation
            )
            plot_bar(
                labels=[x[0] for x in test],
                performance=[x[1] for x in test],
                directory=directory,
                filename='test_distribution_mnb_pos',
                rotation=rotation
            )

            # plot top n words
            plot_bar(
                labels=[x[1] for x in indicative_words['positive']],
                performance=[x[0] for x in indicative_words['positive']],
                directory=directory,
                filename='top_{count}_positive_words_mnb_pos'.format(
                    count=top_words
                ),
                rotation=rotation
            )

            plot_bar(
                labels=[x[1] for x in indicative_words['negative']],
                performance=[x[0] for x in indicative_words['negative']],
                directory=directory,
                filename='top_{count}_negative_words_mnb_pos'.format(
                    count=top_words
                ),
                rotation=rotation
            )

        if kfold:
            kfold_scores['mnb_pos'] = mnb_pos.get_kfold_scores(
                model_type='multinomial',
                n_splits=n_splits
            )

        if prf:
            prf_scores['mnb_pos'] = mnb_pos.get_precision_recall_fscore()

    # bernoulli naive bayes
    if flag_bnb:
        bnb = m_model(
            df=df,
            model_type='bernoulli',
            key_class=key_class,
            key_text=key_text,
            max_length=0,
            ngram=ngram,
            split_size=split_size,
            validate=validate,
            stopwords=stopwords
        )
        model_scores['bnb'] = bnb.get_accuracy()

        if plot:
            plot_cm(
                bnb,
                model_type='bernoulli',
                directory=directory,
                file_suffix='{key_class}{suffix}'.format(
                    key_class=key_class,
                    suffix=suffix
                )
            )

            # extract top n words
            tfidf = bnb.get_tfidf()
            keywords = bnb.get_top_features(
                bnb.get_feature_names(),
                bnb.sort_coo(tfidf.tocoo()),
                top_words
            )

            # sort values: largest to smallest
            keywords = OrderedDict(
                sorted(keywords.items(),
                key=lambda x: x[0])
            )

            # plot top n words
            plot_bar(
                labels=[*keywords],
                performance=[*keywords.values()],
                directory=directory,
                filename='top_{count}_tfidf{suffix}'.format(
                    count=top_words,
                    suffix=suffix
                ),
                rotation=rotation
            )

            # feature distribution
            train = bnb.get_feature_distribution()['y_train']
            test = bnb.get_feature_distribution()['y_test']
            plot_bar(
                labels=[x[0] for x in train],
                performance=[x[1] for x in train],
                directory=directory,
                filename='train_distribution_bnb{suffix}'.format(
                    suffix=suffix
                ),
                rotation=rotation
            )
            plot_bar(
                labels=[x[0] for x in test],
                performance=[x[1] for x in test],
                directory=directory,
                filename='test_distribution_bnb{suffix}'.format(
                    suffix=suffix
                ),
                rotation=rotation
            )

        if kfold:
            kfold_scores['bnb'] = bnb.get_kfold_scores(
                model_type='bernoulli',
                n_splits=n_splits,
                ngram=ngram
            )

        if prf:
            prf_scores['bnb'] = bnb.get_precision_recall_fscore()

    if flag_bnb_pos:
        bnb_pos = mp_model(
            df=df,
            model_type='bernoulli',
            key_class=key_class,
            key_text=key_text,
            max_length=0,
            split_size=split_size,
            validate=validate,
            stopwords=stopwords
        )
        model_scores['bnb_pos'] = bnb_pos.get_accuracy()

        if plot:
            plot_cm(
                bnb_pos,
                model_type='bernoulli',
                directory=directory,
                file_suffix='{}_pos'.format(key_class)
            )

            # extract top n words
            tfidf = bnb_pos.get_tfidf()
            keywords = bnb_pos.get_top_features(
                bnb_pos.get_feature_names(),
                bnb_pos.sort_coo(tfidf.tocoo()),
                top_words
            )

            # sort values: largest to smallest
            keywords = OrderedDict(
                sorted(keywords.items(),
                key=lambda x: x[0])
            )

            # plot top n words
            plot_bar(
                labels=[*keywords],
                performance=[*keywords.values()],
                directory=directory,
                filename='top_{count}_tfidf'.format(count=top_words),
                rotation=rotation
            )

            # feature distribution
            train = bnb_pos.get_feature_distribution()['y_train']
            test = bnb_pos.get_feature_distribution()['y_test']
            plot_bar(
                labels=[x[0] for x in train],
                performance=[x[1] for x in train],
                directory=directory,
                filename='train_distribution_bnb_pos',
                rotation=rotation
            )
            plot_bar(
                labels=[x[0] for x in test],
                performance=[x[1] for x in test],
                directory=directory,
                filename='test_distribution_bnb_pos',
                rotation=rotation
            )

        if kfold:
            kfold_scores['bnb_pos'] = bnb_pos.get_kfold_scores(
                model_type='bernoulli',
                n_splits=n_splits
            )

        if prf:
            prf_scores['bnb_pos'] = bnb_pos.get_precision_recall_fscore()

    # support vector machine
    if flag_svm:
        svm = m_model(
            df=df,
            model_type='svm',
            key_class=key_class,
            key_text=key_text,
            ngram=ngram,
            split_size=split_size,
            validate=validate,
            stopwords=stopwords
        )
        model_scores['svm'] = svm.get_accuracy()

        if plot:
            plot_cm(
                svm,
                model_type='svm',
                directory=directory,
                file_suffix='{key_class}{suffix}'.format(
                    key_class=key_class,
                    suffix=suffix
                )
            )

            # extract top n words
            tfidf = svm.get_tfidf()
            keywords = svm.get_top_features(
                svm.get_feature_names(),
                svm.sort_coo(tfidf.tocoo()),
                top_words
            )

            # sort values: largest to smallest
            keywords = OrderedDict(
                sorted(keywords.items(),
                key=lambda x: x[0])
            )

            # plot top n words
            plot_bar(
                labels=[*keywords],
                performance=[*keywords.values()],
                directory=directory,
                filename='top_{count}_tfidf{suffix}'.format(
                    count=top_words,
                    suffix=suffix
                ),
                rotation=rotation
            )

            # feature distribution
            train = svm.get_feature_distribution()['y_train']
            test = svm.get_feature_distribution()['y_test']
            plot_bar(
                labels=[x[0] for x in train],
                performance=[x[1] for x in train],
                directory=directory,
                filename='train_distribution_svm{suffix}'.format(
                    suffix=suffix
                ),
                rotation=rotation
            )
            plot_bar(
                labels=[x[0] for x in test],
                performance=[x[1] for x in test],
                directory=directory,
                filename='test_distribution_svm{suffix}'.format(
                    suffix=suffix
                ),
                rotation=rotation
            )

        if kfold:
            kfold_scores['svm'] = svm.get_kfold_scores(
                model_type='svm',
                n_splits=n_splits,
                ngram=ngram
            )

        if prf:
            prf_scores['svm'] = svm.get_precision_recall_fscore()

    if flag_svm_pos:
        svm_pos = mp_model(
            df=df,
            model_type='svm',
            key_class=key_class,
            key_text=key_text,
            split_size=split_size,
            validate=validate,
            stopwords=stopwords
        )

        if plot:
            plot_cm(
                svm_pos,
                model_type='svm',
                directory=directory,
                file_suffix='{}_pos'.format(key_class)
            )

            # extract top n words
            tfidf = svm_pos.get_tfidf()
            keywords = svm_pos.get_top_features(
                svm_pos.get_feature_names(),
                svm_pos.sort_coo(tfidf.tocoo()),
                top_words
            )

            # sort values: largest to smallest
            keywords = OrderedDict(
                sorted(keywords.items(),
                key=lambda x: x[0])
            )

            # plot top n words
            plot_bar(
                labels=[*keywords],
                performance=[*keywords.values()],
                directory=directory,
                filename='top_{count}_tfidf'.format(count=top_words),
                rotation=rotation
            )

            # feature distribution
            train = svm_pos.get_feature_distribution()['y_train']
            test = svm_pos.get_feature_distribution()['y_test']
            plot_bar(
                labels=[x[0] for x in train],
                performance=[x[1] for x in train],
                directory=directory,
                filename='train_distribution_svm_pos',
                rotation=rotation
            )
            plot_bar(
                labels=[x[0] for x in test],
                performance=[x[1] for x in test],
                directory=directory,
                filename='test_distribution_svm_pos',
                rotation=rotation
            )

        if kfold:
            kfold_scores['svm_pos'] = svm_pos.get_kfold_scores(
                model_type='svm',
                n_splits=n_splits
            )

        if prf:
            prf_scores['svm_pos'] = svm_pos.get_precision_recall_fscore()

    # ensembled score
    score_good = sum(model_scores.values()) / len(model_scores)

    if plot:
        performance = [x for x in model_scores.values()]
        performance.append(score_good)

        labels = [*model_scores]
        labels.append('overall')

        plot_bar(
            labels=labels,
            performance=performance,
            directory=directory,
            filename='overall_accuracy_{key_class}{suffix}'.format(
                key_class=key_class,
                suffix=suffix
            ),
            rotation=rotation
        )

        [plot_bar(
            labels=range(len(v)),
            performance=v,
            directory=directory,
            filename='bargraph_kfold_{model}{suffix}'.format(
                model=k,
                suffix=suffix
            ),
            rotation=rotation
        ) for k,v in kfold_scores.items()]

        [plot_bar(
            labels=['precision', 'recall', 'fscore'],
            performance=v[:-1],
            directory=directory,
            filename='bargraph_prf_{model}{suffix}'.format(
                model=k,
                suffix=suffix
            ),
            rotation=rotation
        ) for k,v in prf_scores.items()]

    # return score
    return(score_good, kfold_scores, prf_scores, indicative_words)
