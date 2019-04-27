#!/usr/bin/python

from pathlib import Path
from algorithm.text_classifier import Model as alg

def model(
    df=None,
    model_type='multinomial',
    key_text='text',
    key_class='screen_name'
):
    '''

    return trained classifier.

    '''

    # initialize classifier
    if df is not None:
        model = alg(df=df, key_text=key_text, key_class=key_class)
    else:
        model = alg(key_text=key_text, key_class=key_class)

    # vectorize data
    model.split()
    params = model.get_split()
    vectorized = model.get_tfidf()

    # train classifier
    model.train(
        vectorized,
        params['y_train'],
        model_type=model_type
    )

    return(model)

def model_pos(
    m,
    model_type='multinomial',
    key_text='SentimentText',
    key_class='Sentiment'
):
    '''

    return initialized model using pos.

    '''
    # reduce to ascii
    regex = r'[^\x00-\x7f]'
    df_m = m.get_df()
    df_m['pos'] = [re.sub(regex, r' ', sent).split() for sent in df_m['text']]
	
    # suffix pos
    df_m['pos'] = [m.get_pos(x) for x in df_m['pos']]
    m_pos = alg(df=df_m, key_class=key_class, key_text=key_text)

    return(m_pos)
