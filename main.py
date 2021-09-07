import pandas as pd
sp500_0=pd.read_csv('C:/Users/derek/Desktop/ucd/data/all_stocks_5yr_0.csv')
sp500_1=pd.read_csv('C:/Users/derek/Desktop/ucd/data/all_stocks_5yr_1.csv')
#sp500_0=pd.read_csv('https://github.com/derekmcdermott234/UCDPA_DerekMcDermott/upload/master/all_stocks_5yr_0.csv')
#sp500_1=pd.read_csv('https://github.com/derekmcdermott234/UCDPA_DerekMcDermott/upload/master/all_stocks_5yr_1.csv')
print(sp500_0.info())
print(sp500_0.head(5))
print('...')
print(sp500_0.info())
print(sp500_1.head(5))

