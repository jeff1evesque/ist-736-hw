#!/usr/bin/python

import time
import csv
import numpy as np
from pathlib import Path
import pandas as pd


def time_df(fp='{}/data/sample-sentiment.csv'.format(
    Path(__file__).resolve().parents[1]
)):
    '''

    compare upload time for provided csv.

    '''

    # panda
    start_pd = time.time()
    df_pd = pd.read_csv(fp)
    pd_time = time.time() - start_pd

    # numpy + panda
    start_np = time.time()
    with open(fp, 'r') as f:
        data = list(csv.reader(f, delimiter=','))
        df_np = pd.DataFrame(data[1:])
        df_np.columns = data[0]
    np_time = time.time() - start_np

    return({
        'pd_time': pd_time,
        'np_time': np_time,
        'pd_size': df_pd.size,
        'np_size': df_np.size
    })

if __name__ == '__main__':
    tdf = time_df()
    print('panda upload time: {}'.format(tdf['pd_time']))
    print('numpy upload time: {}'.format(tdf['np_time']))
