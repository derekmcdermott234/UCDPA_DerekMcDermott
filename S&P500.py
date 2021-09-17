import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
SP_500 = pd.read_csv('C:/Users/derek/Desktop/ucd/Working Data/sp500.csv',
                   parse_dates=['date'],index_col='date')

print(SP_500.head())
print(SP_500.tail())
print(SP_500.shape)
print(SP_500.info())
print('1',SP_500.isnull().sum())
SP_500_nona=SP_500.fillna(method='ffill')
print('2:',SP_500_nona.isnull().sum())
SP_500_nona_dropped=SP_500_nona.drop_duplicates()
print(SP_500.shape,SP_500_nona_dropped.shape)
#print(nestle.shape,nestle_drop_na.shape)

#nestle_drop_dup_na=nestle_drop_na.drop_duplicates()
#print(nestle.shape,nestle_drop_dup_na.shape)
#nestle_clean=nestle_drop_dup_na
#nestle_close_price=nestle_clean['Close']
#SP500_AAL=SP_500[SP_500['Name']=='AAL']
