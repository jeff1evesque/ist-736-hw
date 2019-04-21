#!/usr/bin/python

#
# this project requires the following packages:
#
#   pip install Twython
#

import sys
sys.path.append('..')
from consumer.twitter_query import TwitterQuery
from config import twitter_api as creds

#
# single query
#
q = TwitterQuery(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])

#
# single query: timeline of screen name.
#
df_musk = q.query_user('elonmusk')
df_jobs = q.query_user('JeffBezos')

#
# analysis
#
print(df_musk)
print(df_jobs)
