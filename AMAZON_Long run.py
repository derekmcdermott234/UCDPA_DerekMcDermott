import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
AMAZ=pd.read_csv('C:/Users/derek/Desktop/ucd/Working Data/yahoo data/AMZN (1).csv',
                   parse_dates=['Date'],index_col='Date')
AMAZ_close_prices=AMAZ['Close']
AMAZ_close_prices=AMAZ_close_prices.loc[:'2021-08-02']
AMAZ_returns=np.array(AMAZ_close_prices.pct_change().dropna()) #calculate the day by day returns on the daily
AMAZ_CLOSE= AMAZ['Close']
AMAZ_CLOSE=AMAZ_CLOSE[5177:].tolist()
print(AMAZ_CLOSE[0])
print(AMAZ_CLOSE[1])
#closing price data and drop the first value which is NaN
Index_1=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44] #create an index for a new dataframe with 10 rows
neww=pd.DataFrame() #A new dataframe is created to store the simulated stock price data
neww['Index']=Index_1
neww=neww.set_index('Index') #set the index of the new dataframe to Index_1 i.e.0 to 9
price_17sep2021 = AMAZ_CLOSE[0]  # store the closing share price on the 18/08/2021 as a constant
print('The price on 17/09/2021 is: ', price_17sep2021)
set_price=np.array([price_17sep2021,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #create a constant list of 20 values to be referenced
print                                                               #within the loop structure to follow.#in one run generate 20 simulated closing prices from 18/08/2021 onwards and add this singe run as a column of data

for l in range(0,1000):    #100 simulations will be run
   y = price_17sep2021    #reset y to the close price on 18/aug/2021
   prices_1=set_price     #reset prices_1 array before it accepts new values from the k indexed loop
   for k in range(0,43):
      y = y+y*(np.random.choice(AMAZ_returns)) #multiply the previous day's closing price by a randomly generated return value
      prices_1[k+1]=y       #store the next day's simulated closing price in a list
   d_series=pd.Series(prices_1) #convert the list to a series before you can add it to neww
   neww=pd.concat([neww,d_series],axis=1)
median_1=neww.median(axis=1)
mean_1=neww.mean(axis=1).tolist()
std_1=neww.std(axis=1).tolist()
mean_1_array=np.array(mean_1)
std_1_array=np.array(std_1)
mean_plus_std_array=mean_1_array+std_1_array
mean_minus_std_array=mean_1_array-std_1_array
mean_plus_2std_array=mean_1_array+2*std_1_array
mean_minus_2std_array=mean_1_array-2*std_1_array
fig,ax=plt.subplots()

plt.plot(median_1,color='b')
plt.plot(mean_1_array,color='c')
plt.plot(mean_plus_std_array,color='r')
plt.plot(mean_minus_std_array,color='r')
plt.plot(mean_plus_2std_array,color='r')
plt.plot(mean_minus_2std_array,color='r')

plt.plot(AMAZ_CLOSE,color='y')
#plt.plot(neww.iloc[:,0:1],color='r')

#---------


plt.show()