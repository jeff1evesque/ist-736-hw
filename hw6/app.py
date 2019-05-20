#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython Quandl wordcloud scikit-plot
#

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from view.exploratory import explore
from view.classifier import plot_bar
from exploratory.sentiment import Sentiment
from datetime import datetime
from controller.classifier import classify
import matplotlib.pyplot as plt

#
# read dataset: sample equally 500 by gender.
#
df = pd.read_csv('{}/data/380000-lyrics-from-metrolyrics/lyrics1.csv'.format(
    Path(__file__).resolve().parents[1]
))
df = df.groupby('genre').apply(lambda x: x.sample(500))

if not os.path.exists('viz'):
    os.makedirs('viz')

#
# classify
#
classify_results[sn] = classify(
    df,
    key_class='genre',
    key_text='lyrics',
    directory='viz',
    top_words=25
)
