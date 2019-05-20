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
    n_splits=5,
    top_words=20,
    ngram_range=(1,1),
    rotatio=90,
    directory='viz',
    flag_mnb=True,
    flag_mnb_pos=True,
    flag_bnb=True,
    flag_bnb_pos=True,
    flag_svm=True,
    flag_svm_pos=True,
    plot=True
):
    '''

    implement designated classifiers.

    '''

    # local variables
    kfold_scores = {}
    model_scores = {}

    if ngram_range == (1,1):
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
            ngram=ngram
        )
        model_scores['mnb'] = mnb.get_accuracy()

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

        if kfold:
            kmnb = mnb.get_kfold_scores(
                model_type='multinomial',
                n_splits=n_splits,
                ngram=ngram
            )
            kfold_scores['mnb'] = kmnb

    if flag_mnb_pos:
        mnb_pos = mp_model(
            df=df,
            key_class=key_class,
            key_text=key_text,
            max_length=math.inf
        )
        model_scores['mnb_pos'] = mnb_pos.get_accuracy()

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

        if kfold:
            kmnb_pos = mnb_pos.get_kfold_scores(
                model_type='multinomial',
                n_splits=n_splits
            )
            kfold_scores['mnb_pos'] = kmnb_pos

    # bernoulli naive bayes
    if flag_bnb:
        bnb = m_model(
            df=df,
            model_type='bernoulli',
            key_class=key_class,
            key_text=key_text,
            max_length=0,
            ngram=ngram
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
            kbnb = bnb.get_kfold_scores(
                model_type='bernoulli',
                n_splits=n_splits,
                ngram=ngram
            )
            kfold_scores['bnb'] = kbnb

    if flag_bnb_pos:
        bnb_pos = mp_model(
            df=df,
            model_type='bernoulli',
            key_class=key_class,
            key_text=key_text,
            max_length=0
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
            kbnb_pos = bnb_pos.get_kfold_scores(
                model_type='bernoulli',
                n_splits=n_splits
            )
            kfold_scores['bnb_pos'] = kbnb_pos

    # support vector machine
    if flag_svm:
        svm = m_model(
            df=df,
            model_type='svm',
            key_class=key_class,
            key_text=key_text,
            ngram=ngram
        )
        model_scores['svm'] = svm.get_accuracy()

        if plot:
            plot_cm(
                svm,
                model_type='svm',
                directory=directory,
                file_suffix='{key_class}_pos{suffix}'.format(
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
            ksvm = svm.get_kfold_scores(
                model_type='svm',
                n_splits=n_splits,
                ngram=ngram
            )
            kfold_scores['svm'] = ksvm

    if flag_svm_pos:
        svm_pos = mp_model(
            df=df,
            model_type='svm',
            key_class=key_class,
            key_text=key_text
        )
        model_scores['svm_pos'] = svm_pos.get_accuracy()

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
            ksvm_pos = svm_pos.get_kfold_scores(
                model_type='svm',
                n_splits=n_splits
            )
            kfold_scores['svm_pos'] = ksvm_pos

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
            range(len(v)),
            v,
            directory=directory,
            filename='bargraph_kfold_{model}{suffix}'.format(
                model=k,
                suffix=suffix
            ),
            rotation=rotation
        ) for k,v in kfold_scores.items()]

    # return score
    return(score_good, kfold_scores)
