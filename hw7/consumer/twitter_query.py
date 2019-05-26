#!/usr/bin/python

import re
import json
from twython import Twython, TwythonError
import pandas as pd
from datetime import datetime


class TwitterQuery():
    '''

    implements twitter api.

    '''

    def __init__(self, key, secret):
        '''

        define class variables.

        '''

        self.regex = r'[^\x00-\x7f]'
        self.conn = Twython(key, secret)

    def get_dict_val(self, d, keys):
        '''

        return value of nested dict using provided list of keys.

        '''

        for k in keys:
            d = d.get(k, None)
            if d is None:
                return(None)

        if isinstance(d, dict) or isinstance(d, list):
            return(json.dumps(d))
        else:
            return(d)

    def get_dict_path(self, d):
        '''

        given a dict:

            d = {
                'A': {
                    'leaf-1': 'value-1',
                    'node-1': {'leaf-sub-1': 'sub-value-1'}
                }
            }

        a nested list is generated and returned:

            [
                ['A', 'leaf-1', 'value-1'],
                ['A', 'node-1', 'leaf-sub-1', 'sub-value-1']
            ]

        '''

        temp = []
        result = []
        for i,(k,v) in enumerate(d.items()):
            if isinstance(v, dict):
                temp.append(k)
                self.get_dict_path(v)

            else:
                if isinstance(v, list):
                    temp.append(k)
                    [temp.append(x) for x in v]
                    result.append(temp)
                    temp = []

                else:
                    temp.append(k)
                    result.append(temp)
                    temp = []

        return(result)

    def query(
        self,
        query,
        params=[{'user': ['screen_name']}, 'created_at', 'full_text'],
        sorted=None,
        force_ascii=True
    ):
        '''

        search tweets using provided parameters and default credentials.

        @query, query parameters of the form:

            {
                'q': 'learn python',  
                'result_type': 'popular',
                'count': 10,
                'lang': 'en',
            }

        @params, parameters to return.
        @keys, list of lists, recursive params key through end value.

        Note: additional search arguments, as well as full response
              'statuses' can be utilized and referenced:

            - https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets.html

        '''

        keys = []
        [keys.extend(self.get_dict_path(k)) if isinstance(k, dict) else keys.append([k]) for k in params]
        results = {x[-1]: [] for x in keys}

        #
        # query
        #
        for status in self.conn.search(**query)['statuses']:
            [results[k].append(self.get_dict_val(status, keys[i])) for i, (k,v) in enumerate(results.items())]

        self.df_query = pd.DataFrame(results)

        if force_ascii:
            self.df_timeline['full_text'] = [re.sub(self.regex, r' ', str(s)) for s in self.df_timeline['full_text']]

        return(self.df_query)

    def query_user(
        self,
        screen_name,
        params=[{'user': ['screen_name']}, 'created_at', 'full_text'],
        count=200,
        rate_limit=0,
        force_ascii=True
    ):
        '''

        search tweets by supplied screen name.

        @screen_name, user timeline to query.
        @rate_limit, number of api request (limited at 900), where each request
            has a maximum count=200 tweets to return.
        @params, parameters to return.
        @count, number of tweets to return.
        @keys, list of lists, recursive params key through end value.

        '''

        # local variables
        if rate_limit > 900:
            rate_limit = 900

        elif rate_limit < 0:
            rate_limit = 0

        #
        # query: induction step
        #
        try:
            timeline = self.conn.get_user_timeline(
                screen_name=screen_name,
                count=count,
                tweet_mode='extended'
            )

        except TwythonError as e:
            print(e)

        keys = []
        [keys.extend(self.get_dict_path(k)) if isinstance(k, dict) else keys.append([k]) for k in params]

        results = {}
        for i,x in enumerate(keys):
            if x[-1] in results:
                results['{x}_{suffix}'.format(x=x[-1], suffix=i)] = []
            else:
                results[x[-1]] = []

        for tweet in timeline:
            last_id = tweet['id']
            [results[k].append(self.get_dict_val(
                tweet,
                keys[i]
            )) for i,(k,v) in enumerate(results.items())]

        #
        # query: extend through max limit
        #
        for i in range(rate_limit):
            new_timeline = self.conn.get_user_timeline(
                screen_name = screen_name,
                count = count,
                tweet_mode='extended',
                max_id = last_id - 1
            )

            if len(new_timeline) > 0:
                for tweet in new_timeline:
                    last_id = tweet['id']
                    [results[k].append(self.get_dict_val(
                        tweet,
                        keys[i]
                    )) for i,(k,v) in enumerate(results.items())]

        #
        # store results
        #
        self.df_timeline = pd.DataFrame(results)

        #
        # clean dataframe
        #
        if force_ascii:
            self.df_timeline['full_text'] = [re.sub(self.regex, r' ', str(s)) for s in self.df_timeline['full_text']]

        # reformat date
        self.df_timeline['created_at'] = [datetime.strptime(
            x,
            '%a %b %d %H:%M:%S +0000 %Y'
        ) for x in self.df_timeline['created_at']]

        # duplicate index column: side effect of datetime conversion.
        self.df_timeline.drop(self.df_timeline.filter(regex='Unnamed:').columns, axis=1, inplace=True)
        self.df_timeline.sort_values(by='created_at', ascending=True, inplace=True)
        self.df_timeline.reset_index(inplace=True)
        self.df_timeline.drop(['index'], axis=1, inplace=True)

        return(self.df_timeline)
