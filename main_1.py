import pandas as pd
import numpy as np
sp500_0=pd.read_csv('C:/Users/derek/Desktop/ucd/data/all_stocks_5yr_0.csv',parse_dates=True,index_col='date')
sp500_1=pd.read_csv('C:/Users/derek/Desktop/ucd/data/all_stocks_5yr_1.csv',parse_dates=True,index_col='date')
#sp500_0=pd.read_csv('https://github.com/derekmcdermott234/UCDPA_DerekMcDermott/upload/master/all_stocks_5yr_0.csv')
#sp500_1=pd.read_csv('https://github.com/derekmcdermott234/UCDPA_DerekMcDermott/upload/master/all_stocks_5yr_1.csv')
print('...............')
print(sp500_0.shape)
print(sp500_0.head(5))
missing_values_count_0=sp500_0.isnull().sum()
print(missing_values_count_0[:])
null_data_0 = sp500_0[sp500_0.isnull().any(axis=1)]
print(null_data_0)
droprows_0=sp500_0.dropna()
print(sp500_0.shape,droprows_0.shape)

print('...............')
print(sp500_1.shape)
print(sp500_1.head(5))
missing_values_count_1=sp500_1.isnull().sum()
print(missing_values_count_1[:])
null_data_1 = sp500_1[sp500_1.isnull().any(axis=1)]
print(null_data_1)
droprows_1=sp500_1.dropna()
print(sp500_1.shape,droprows_1.shape)

