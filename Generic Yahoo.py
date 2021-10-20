import pandas as pd
from yahoofinancials import YahooFinancials
yahoo_financials = YahooFinancials('TSLA')

data = yahoo_financials.get_historical_price_data(start_date='2000-01-01',
                                                  end_date='2019-12-31',
                                                  time_interval='weekly')

tsla_df = pd.DataFrame(data['TSLA']['prices'])
tsla_df = tsla_df.drop('date', axis=1).set_index('formatted_date')
tsla_df.head()
print(tsla_df.head())