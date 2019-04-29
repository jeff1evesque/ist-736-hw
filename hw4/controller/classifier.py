#!/usr/bin/python

import math
from model.classifier import model as m_model
from model.classifier import model_pos as mp_model
from view.classifier import plot_cm, plot_bar

def classify(
    df,
    key_class='screen_name',
    key_text='text',
    kfold=True,
    n_splits=5,
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

    # multinomial naive bayes
    if flag_mnb:
        mnb = m_model(
            df=df,
            key_class=key_class,
            key_text=key_text,
            max_length=math.inf
        )
        model_scores['mnb'] = mnb.get_accuracy()

        if plot:
            plot_cm(mnb, file_suffix=key_class)

        if kfold:
            kmnb = mnb.get_kfold_scores(model_type='multinomial', n_splits=n_splits)
            kfold_scores['mnb'] = kmnb

    if flag_mnb_pos:
        mnb_pos = mp_model(
            mnb,
            key_class=key_class,
            key_text=key_text,
            max_length=math.inf
        )
        model_scores['mnb_pos'] = mnb_pos.get_accuracy()

        if plot:
            plot_cm(mnb_pos, file_suffix='{}_pos'.format(key_class))

        if kfold:
            kmnb_pos = mnb_pos.get_kfold_scores(model_type='multinomial', n_splits=n_splits)
            kfold_scores['mnb_pos'] = kmnb_pos

    # bernoulli naive bayes
    if flag_bnb:
        bnb = m_model(
            df=df,
            model_type='bernoulli',
            key_class=key_class,
            key_text=key_text,
            max_length=0
        )
        model_scores['bnb'] = bnb.get_accuracy()

        if plot:
            plot_cm(bnb, model_type='bernoulli', file_suffix=key_class)

        if kfold:
            kbnb = bnb.get_kfold_scores(model_type='bernoulli', n_splits=n_splits)
            kfold_scores['bnb'] = kbnb

    if flag_bnb_pos:
        bnb_pos = mp_model(
            bnb,
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
                file_suffix='{}_pos'.format(key_class)
            )

        if kfold:
            kbnb_pos = bnb_pos.get_kfold_scores(model_type='bernoulli', n_splits=n_splits)
            kfold_scores['bnb_pos'] = kbnb_pos

    # support vector machine
    if flag_svm:
        svm = m_model(
            df=df,
            model_type='svm',
            key_class=key_class,
            key_text=key_text
        )
        model_scores['svm'] = svm.get_accuracy()

        if plot:
            plot_cm(svm, model_type='svm', file_suffix=key_class)

        if kfold:
            ksvm = svm.get_kfold_scores(model_type='svm', n_splits=n_splits)
            kfold_scores['svm'] = ksvm

    if flag_svm_pos:
        svm_pos = mp_model(
            svm,
            model_type='svm',
            key_class=key_class,
            key_text=key_text
        )
        model_scores['svm_pos'] = svm_pos.get_accuracy()

        if plot:
            plot_cm(
                svm_pos,
                model_type='svm',
                file_suffix='{}_pos'.format(key_class)
            )

        if kfold:
            ksvm_pos = svm_pos.get_kfold_scores(model_type='svm', n_splits=n_splits)
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
            filename='overall_accuracy_{}'.format(key_class)
        )

    # return score
    return(score_good, kfold_scores)
