import pandas as pd
import numpy as np
sp500_0=pd.read_csv('C:/Users/derek/Desktop/ucd/data/all_stocks_5yr_0.csv',parse_dates=True,index_col='date')
sp500_1=pd.read_csv('C:/Users/derek/Desktop/ucd/data/all_stocks_5yr_1.csv',parse_dates=True,index_col='date')
#sp500_0=pd.read_csv('https://github.com/derekmcdermott234/UCDPA_DerekMcDermott/upload/master/all_stocks_5yr_0.csv')
#sp500_1=pd.read_csv('https://github.com/derekmcdermott234/UCDPA_DerekMcDermott/upload/master/all_stocks_5yr_1.csv')
sp_500=pd.concat([sp500_0,sp500_1],axis=0)
sp_500_drop_na=sp_500.dropna()
print(sp_500.shape,sp_500_drop_na.shape)
sp_500_drop_dup_and_na=sp_500_drop_na.drop_duplicates()
print(sp_500_drop_na.shape,sp_500_drop_dup_and_na.shape)
print(sp_500_drop_dup_and_na.isna().sum())
print(sp_500_drop_dup_and_na[sp_500_drop_dup_and_na.duplicated()])