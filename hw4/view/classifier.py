#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

def plot_cm(
    model,
    model_type='multinomial',
    file_suffix='text',
):
    '''

    plot confusion matrix.

    '''

    model.plot_cm(filename='viz/cm_{m}_{s}.png'.format(
        m=model_type,
        s=file_suffix
    ))

def plot_bar(
    labels,
    performance,
    model_type='multinomial',
    file_suffix='text',
):
    '''

    plot confusion matrix.

    '''

    y_pos = np.arange(len(labels))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Performance')
    plt.savefig('viz/accuracy_overall.png')
    plt.show()
