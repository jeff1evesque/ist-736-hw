#!/usr/bin/python

from twython import Twython, TwythonError
import pandas as pd


class TwitterQuery():     
    '''

    requires instance of TwitterStream.

    '''

    def __init__(self, key, secret):
        '''

        define class variables.

        '''

        self.conn = Twython(key, secret)

    def get_dict_val(self, d, keys):
        '''

        return value of nested dict using provided list of keys.

        '''

        for k in keys:
            d = d.get(k, None)
            if d is None:
                return(None)
        return(d.encode('utf-8'))

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
        for k,v in d.items():
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
                    temp.append(v)
                    result.append(temp)
                    temp = []

        return(result)

    def query(
        self,
        query,
        params=[{'user': ['screen_name']}, 'created_at', 'text'],
        sorted=None
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
        # Note: to debug within the loop, and access 'status':
        #
        #       print(repr(status).encode('utf-8'))
        #
        for status in self.conn.search(**query)['statuses']:
            [results[k].append(self.get_dict_val(status, keys[i])) for i, (k,v) in enumerate(results.items())]

        self.df_query = pd.DataFrame(results)
        return(self.df_query)

    def query_user(
        self,
        screen_name,
        params=[{'user': ['screen_name']}, 'created_at', 'text'],
        count=200
    ):
        '''

        search tweets by supplied screen name.

        @screen_name, user timeline to query.
        @params, parameters to return.
        @count, number of tweets to return.
        @keys, list of lists, recursive params key through end value.

        '''

        try:
            timeline = self.conn.get_user_timeline(screen_name=screen_name, count=count)
        except TwythonError as e:
            print(e)

        keys = []
        [keys.extend(self.get_dict_path(k)) if isinstance(k, dict) else keys.append([k]) for k in params]
        results = {x[-1]: [] for x in keys}

        #
        # query
        #
        # Note: to debug within the loop, and access 'status':
        #
        #       print(repr(tweet).encode('utf-8'))
        #
        for tweet in timeline:
            [results[k].append(self.get_dict_val(tweet, keys[i])) for i, (k,v) in enumerate(results.items())]

        self.df_timeline = pd.DataFrame(results)
        return(self.df_timeline)
