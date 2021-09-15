import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
fig,ax=plt.subplots(3,3)
nestle=pd.read_csv('C:/Users/derek/Desktop/ucd/Working Data/yahoo data/NESN.SW.csv',
                   parse_dates=['Date'],index_col='Date')
print(nestle.shape)
nestle.info()
print(nestle.head())
missing_values_count=nestle.isnull().sum()
print(missing_values_count)
nestle_drop_na=nestle.dropna()
print(nestle.shape,nestle_drop_na.shape)
nestle_drop_dup_na=nestle_drop_na.drop_duplicates()
print(nestle.shape,nestle_drop_na.shape,nestle_drop_dup_na.shape)
nestle_clean=nestle_drop_dup_na
nestle_close_price=nestle_clean['Close']
nestle_int_check=nestle_close_price.astype(int)
print(nestle_int_check[nestle_int_check==0])
print(nestle_close_price.head())
nestle_daily_returns=nestle_close_price.pct_change().dropna()
print(nestle_daily_returns.head())
z=0
j=0
price_17aug2021=nestle_close_price[2728]
set_price=[price_17aug2021,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
mu = nestle_daily_returns[:2729].mean() #to 17/08/2021
standard_dev=nestle_daily_returns[:2729].std()
print('...')
print(mu,standard_dev)
print('...')
Index_1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
neww=pd.DataFrame()
neww['Index']=Index_1
neww=neww.set_index('Index')
neww['new_col'] = np.array(nestle_close_price.iloc[2728:2748])
phi= np.random.normal(loc=mu, scale=standard_dev, size=(1,1000))
for l in range(0,100):
   y = price_17aug2021
   prices_1=set_price
   for k in range(0,19):
      y = y+y*(phi[0][l+k])
      prices_1[k+1]=y
   d_series=pd.Series(prices_1)
   neww=pd.concat([neww,d_series],axis=1)
c_neww=neww.drop("new_col",axis=1)
c_neww['MEAN']=c_neww.mean(axis=1)
c_neww['MEAN'].to_csv('C:/Users/derek/Desktop/ucd/Working Data/yahoo data/nestle_mean_columns.csv',mode='a', index=True, header=False)
print(c_neww['MEAN'].head())
print(c_neww.info())
print('...')
print(neww.info())
print(neww)
print('...')
sim_2=neww.iloc[:,1:2];sim_3=neww.iloc[:,2:3];sim_4=neww.iloc[:,3:4];sim_5=neww.iloc[:,4:5]
sim_6=neww.iloc[:,5:6];sim_7=neww.iloc[:,6:7];sim_8=neww.iloc[:,7:8];sim_9=neww.iloc[:,8:9]
ax[0][0].plot(neww['new_col'],color='b')
ax[0][0].plot(c_neww['MEAN'],color='r')
ax[0][1].plot(neww['new_col'],color='b')
ax[0][1].plot(sim_2,color='r')
ax[0][2].plot(neww['new_col'],color='b')
ax[0][2].plot(sim_3,color='r')
ax[1][0].plot(neww['new_col'],color='b')
ax[1][0].plot(sim_4,color='r')
ax[1][1].plot(neww['new_col'],color='b')
ax[1][1].plot(sim_5,color='r')
ax[1][2].plot(neww['new_col'],color='b')
ax[1][2].plot(sim_6,color='r')
ax[2][0].plot(neww['new_col'],color='b')
ax[2][0].plot(sim_7,color='r')
ax[2][1].plot(neww['new_col'],color='b')
ax[2][1].plot(sim_8,color='r')
ax[2][2].plot(neww['new_col'],color='b')
ax[2][2].plot(sim_9,color='r')
plt.show()
print(neww['new_col'])