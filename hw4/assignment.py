#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython
#

import sys
sys.path.append('..')
import pandas as pd
from pathlib import Path
from model.text_classifier import Model as model
from utility.dataframe import standardize_df

# local variables
adjusted_csv = 'adjusted_data.csv'

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

lie = model(
    df=df_adjusted,
    key_text='review',
    key_class='lie'
)
sentiment = model(
    df=df_adjusted,
    key_text='review',
    key_class='sentiment'
)

# normalize labels
df_lie = lie.get_df()
df_sentiment = sentiment.get_df()

df_lie = df_lie.replace({'lie': {'f': 0, 't': 1}})
df_sentiment = df_sentiment.replace({'lie': {'n': 0, 'p': 1}})

print(df_lie)
