#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

def plot_cm(
    model,
    directory='viz',
    model_type='multinomial',
    file_suffix='text'
):
    '''

    plot confusion matrix.

    '''

    model.plot_cm(filename='{d}/cm_{m}_{s}.png'.format(
        d=directory,
        m=model_type,
        s=file_suffix
    ))

def plot_bar(
    labels,
    performance,
    directory='viz',
    filename='text',
    rotation=0,
    show=False
):
    '''

    plot confusion matrix.

    '''

    y_pos = np.arange(len(labels))
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, labels, rotation=rotation)
    plt.ylabel('Performance')
    plt.savefig('{d}/{f}'.format(d=directory, f=filename))

    if show:
        plt.show()
    else:
        plt.close()
