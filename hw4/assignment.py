#!/usr/bin/python

import sys
sys.path.append('..')
import pandas as pd
from pathlib import Path
from utility.dataframe import standardize_df
from model.classifier import model as m_model
from model.classifier import model_pos as mp_model
from view.classifier import plot_cm, plot_bar

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

# normalize labels
df_adjusted = df_adjusted.replace({
    'lie': {'f': 0, 't': 1},
    'sentiment': {'n': 0, 'p': 1}
})

#
# unigram lie detection
#

# multinomial naive bayes
m_mnb = m_model(key_class='lie', key_text='review')
m_mnb_accuracy = m_mnb.get_accuracy()
plot_cm(m_mnb, key_class='lie', key_text='review', file_suffix='lie')

m_mnb_pos = mp_model(key_class='lie', key_text='review')
m_mnb_pos_accuracy = m_mnb_pos.get_accuracy()
plot_cm(m_mnb_pos, key_class='lie', key_text='review', file_suffix='lie_pos')

# bernoulli naive bayes
m_bnb = m_model(model_type='bernoulli', key_class='lie', key_text='review')
m_bnb_accuracy = m_bnb.get_accuracy()
plot_cm(m_mnb, model_type='bernoulli', key_class='lie', key_text='review', file_suffix='lie')

m_bnb_pos = mp_model(model_type='bernoulli', key_class='lie', key_text='review')
m_bnb_pos_accuracy = m_bnb_pos.get_accuracy()
plot_cm(m_mnb_pos, model_type='bernoulli', key_class='lie', key_text='review', file_suffix='lie_pos')

# support vector machine
m_svm = m_model(model_type='svm', key_class='lie', key_text='review')
m_svm_accuracy = m_svm.get_accuracy()
plot_cm(m_svm, model_type='svm', key_class='lie', key_text='review', file_suffix='lie')

m_svm_pos = mp_model(model_type='svm', key_class='lie', key_text='review')
m_svm_pos_accuracy = m_svm_pos.get_accuracy()
plot_cm(m_svm_pos, model_type='svm', key_class='lie', key_text='review', file_suffix='lie_pos')

# ensembled score
score_good = (m_mnb_accuracy + m_mnb_pos_accuracy + m_bnb_accuracy + m_bnb_pos_accuracy + m_svm_accuracy + m_svm_pos_accuracy) / 6
score_bad = 1 - score_good
plot_bar(
    labels=('mnb', 'mnb pos', 'bnb', 'bnb pos', 'svm', 'svm pos', 'overall'),
    performance=(m_mnb_accuracy, m_mnb_pos_accuracy, m_bnb_accuracy, m_bnb_pos_accuracy, m_svm_accuracy, m_svm_pos_accuracy, score_good)
)

#
# unigram sentiment analysis
#
print(df_lie)
