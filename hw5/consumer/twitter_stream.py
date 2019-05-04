#!/usr/bin/python

import csv
from twython import Twython  
from twython import TwythonStreamer


class TwitterStream(TwythonStreamer):     
    '''

    TwythonStreamer subclass.

    '''

    def on_success(self, data):
        '''

        required 'TwythonStreamer' method called when twitter returns data. 

        '''

        tweet_data = process_tweet(data)
        self.save_to_csv(tweet_data)

    def on_error(self, status_code, data):
        '''

        required 'TwythonStreamer' method called when twitter returns an error. 

        '''

        print(status_code, data)
        self.disconnect()

    def save_to_csv(self, tweet):
        '''

        optional 'TwythonStreamer' method to store tweets into a file. 

        '''

        with open(r'saved_tweets.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(list(tweet.values()))
