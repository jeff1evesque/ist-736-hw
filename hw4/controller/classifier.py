#!/usr/bin/python

import math
from model.classifier import model as m_model
from model.classifier import model_pos as mp_model
from view.classifier import plot_cm, plot_bar

def classify(
    df,
    key_class='screen_name',
    key_text='text',
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

    # multinomial naive bayes
    if flag_mnb:
        mnb = m_model(
            df=df,
            key_class=key_class,
            key_text=key_text,
            max_length=math.inf
        )
        mnb_accuracy = mnb.get_accuracy()

        if plot:
            plot_cm(mnb, file_suffix=key_class)

    if flag_mnb_pos:
        mnb_pos = mp_model(
            mnb,
            key_class=key_class,
            key_text=key_text,
            max_length=math.inf
        )
        mnb_pos_accuracy = mnb_pos.get_accuracy()

        if plot:
            plot_cm(lie_mnb_pos, file_suffix='{}_pos'.format(key_class))

    # bernoulli naive bayes
    if flag_bnb:
        bnb = m_model(
            df=df,
            model_type='bernoulli',
            key_class=key_class,
            key_text=key_text,
            max_length=0
        )
        bnb_accuracy = bnb.get_accuracy()

        if plot:
            plot_cm(bnb, model_type='bernoulli', file_suffix=key_class)

    if flag_bnb_pos:
        bnb_pos = mp_model(
            bnb,
            model_type='bernoulli',
            key_class=key_class,
            key_text=key_text,
            max_length=0
        )
        bnb_pos_accuracy = bnb_pos.get_accuracy()

        if plot:
            plot_cm(
                bnb_pos,
                model_type='bernoulli',
                file_suffix='{}_pos'.format(key_class)
            )

    # support vector machine
    if flag_svm:
        svm = m_model(
            df=df,
            model_type='svm',
            key_class=key_class,
            key_text=key_text
        )
        svm_accuracy =  svm.get_accuracy()

        if plot:
            plot_cm(svm, model_type='svm', file_suffix=key_class)

    if flag_svm_pos:
        svm_pos = mp_model(
            svm,
            model_type='svm',
            key_class=key_class,
            key_text=key_text
        )
        svm_pos_accuracy = svm_pos.get_accuracy()

        if plot:
            plot_cm(
                svm_pos,
                model_type='svm',
                file_suffix='{}_pos'.format(key_class)
            )

    # ensembled score
    score_good = (\
        mnb_accuracy + \
        mnb_pos_accuracy + \
        bnb_accuracy + \
        bnb_pos_accuracy + \
        svm_accuracy + \
        svm_pos_accuracy\
    ) / 6

    if plot:
        plot_bar(
            labels=('mnb', 'mnb pos', 'bnb', 'bnb pos', 'svm', 'svm pos', 'overall'),
            performance=(
                mnb_accuracy,
                mnb_pos_accuracy,
                bnb_accuracy,
                bnb_pos_accuracy,
                svm_accuracy,
                svm_pos_accuracy,
                score_good
            ),
            filename='overall_accuracy_{}'.format(key_class)
        )

    # return score
    return(score_good)
