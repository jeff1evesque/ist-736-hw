#!/usr/bin/python

import quandl


class QuandlQuery():
    '''

    query against specified quandl database. Authenticated requests with an apikey have
    the following api limits:

        - 300 per 10 seconds
        - 2000 per 10 minutes
        - 50000 per day

    '''

    def __init__(self, apikey):
        '''

        define class variables.

        '''

        self.quandl = quandl
        self.quandl.ApiConfig.api_key = apikey

    def get_ts(
        self,
        database_code='NASDAQOMX',
        dataset_code='COMP-NASDAQ',
        start_date=None,
        end_date=None,
        collapse='daily'
    ):
        '''

        return timeseries data.

        @database_code, where the data comes from (quandl as over 500 databases).
        @dataset_code, specific time series (i.e. FB, APPL).
        @start_date, retrieve data rows on and after the specified start date.
        @end_date, retrieve data rows up to and including the specified end date.
        @collapse, aggregate date into segments (i.e. 'monthly').

        Note: xxx_date must be of the 'yyyy-mm-dd' format.

        '''

        if start_date and end_date:
            self.query = self.quandl.get(
                '{s}/{t}'.format(s=database_code, t=dataset_code),
                collapse=collapse,
                start_date=start_date,
                end_date=end_date
            )

        elif start_date:
            self.query = result = self.quandl.get(
                '{s}/{t}'.format(s=database_code, t=dataset_code),
                collapse=collapse,
                start_date=start_date
            )

        elif end_date:
            self.query = self.quandl.get(
                '{s}/{t}'.format(s=database_code, t=dataset_code),
                collapse=collapse,
                end_date=end_date
            )

        else:
            self.query = self.quandl.get(
                '{s}/{t}'.format(s=database_code, t=dataset_code),
                collapse=collapse
            )

        return(self.query)
