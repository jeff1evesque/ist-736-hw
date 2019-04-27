#!/usr/bin/python

import pandas as pd

def standardize_df(fp):
    '''

    generate dataframe from csv containing unbalanced columns.

    '''

    # local variables
    delimiter = ','
    largest_column_count = 0

    # determine max column
    with open(fp, 'r') as f:
        lines = f.readlines()

        for l in lines:
            column_count = len(l.split(delimiter)) + 1

            if largest_column_count < column_count:
                largest_column_count = column_count

    # close file
    f.close()

    # temporary dataframe: allows uneven columns
    column_names = ['col-{}'.format(i) for i in range(largest_column_count)]
    return(pd.read_csv(fp, header=None, delimiter=delimiter, names=column_names))
