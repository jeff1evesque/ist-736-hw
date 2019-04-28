#!/usr/bin/python

import os
import sys
sys.path.append('..')
import pandas as pd
from pathlib import Path
from controller.classifier import classify

# local variables
adjusted_csv = 'adjusted_data.csv'

# ensure directory
if not os.path.exists('viz'):
    os.makedirs('viz')

if not Path(adjusted_csv).is_file():
    # load unbalanced data
    df = standardize_df('{}/data/deception_data_converted_final.csv'.format(
        Path(__file__).resolve().parents[1]
    ))

    #
    # merge unbalanced 'review' columns into one column
    #
    col_names = list(df)
    df_adjusted = pd.DataFrame({
        'lie': df.iloc[:,0],
        'sentiment': df.iloc[:,1]
    }).sort_index()

    review_columns = [x for x in col_names[2:]]
    df_adjusted['review'] = df[review_columns].apply(lambda row: ' '.join(
        row.values.astype(str)),
        axis=1
    )
    df_adjusted.to_csv(adjusted_csv)

else:
    df_adjusted = pd.read_csv(adjusted_csv)

# normalize labels
df_adjusted = df_adjusted.replace({
    'lie': {'f': 0, 't': 1},
    'sentiment': {'n': 0, 'p': 1}
})

#
# unigram lie detection
#
classify(df_adjusted, key_class='lie', key_text='review')

#
# unigram sentiment analysis
#
classify(df_adjusted, key_class='sentiment', key_text='review')
