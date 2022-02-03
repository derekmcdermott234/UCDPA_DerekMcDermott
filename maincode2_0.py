import numpy as np
import pandas as pd
import yfinance as yf
nestle = yf.download(tickers="SPCE", period="5d", interval="1m")
print(nestle.shape)              #print the shape of the dataframe nestle
print(nestle.info())             #print information on the dataframe nestle
print(nestle.head())             #Show the first 5 rows of nestle
print(nestle.isnull().sum())     #print the missing values count for each column
nestle_drop_na=nestle.dropna()   #drop missing values from nestle and store as a new dataframe nestle_drop_na
print(nestle.shape,nestle_drop_na.shape)  #print the shape of the dataframe with dropped n/a's compared to the
                                          # original nestle data frame
nestle_drop_dup_na=nestle_drop_na.drop_duplicates() #also drop rows that are duplicate and store as new dataframe
                                                    #nestle_drop_dup_na
print(nestle.shape,nestle_drop_dup_na.shape)        #compare original nestle dataframe to cleaned dataframe
nestle_clean=nestle_drop_dup_na                     #rename the cleaned dataframe as nestle_clean
nestle_close_price=nestle_clean['Close']            #get the Close price column from nestle_clean to a new
                                                    #series 'nestle_close_price'
def check_for_zeros(x):                    #define function to check for zeros in a series or list
   int_check=x.astype(int)                 #convert floating point data in passed series to integer type data
   return int_check[int_check==0].count()  #count the zeros in the passed list and return the result.
print(check_for_zeros(nestle_close_price))     #call on the function to check nestle_close_price data for zeros
print(nestle_close_price.head()) #print the nestle close price data to inspect
nestle_daily_returns=nestle_close_price.pct_change().dropna() #calculate the day by day returns on the daily
                                                              #closing price data and drop the first value which is NaN
mu = nestle_daily_returns.iloc[-420].mean() #calcullate the mean of the daily returns up to 18/08/2021
standard_dev=nestle_daily_returns.iloc[-420].std() #calcullate the standard deviation of the daily returns up to 17/08/2021
print('...')
print(mu,standard_dev) #print the mean and standard deviation of the daily returns series
print('...')
I_list=[0]*420
p=0
for p in range(0,420):
    I_list[p]=p
print(I_list)
Index_1=I_list #A list is created that will later index a data frame with 11 rows
RESAMP=pd.DataFrame() #Two new dataframes are created that will later store simulated and
                                          #indexed stock price paths column by column
                                          #NORM will store Gaussian generated stock prices based on historical mean and std
                                          #RESAMP will store stock prices based on historical data that is resampled and replaced
RESAMP=RESAMP.reset_index() #Reset the index for both data frames
RESAMP['Index']=Index_1                  #Put in a new integer index in both dataframes
RESAMP=RESAMP.set_index('Index')  ##
RESAMP=RESAMP.drop('index', axis=1) #drop the old index which is stored in an 'index' column
print('**********************************************************')
print(RESAMP.shape)
print(RESAMP.head())
print(RESAMP.info())
 #generate an array of 1x15000 random numbers picked
                                                                  #from a gaussian distribution with a mean of 'mu' and
                                                                  # a standard deviation of 'standard_dev'

NESTLE_CLOSE= np.array(nestle_close_price.iloc[-420:-1])     #get 11 nestle close prices from 18/08/2021
                                                               # plus 10 days when stock market is open

nestle_close_price_index=nestle_close_price.iloc[-420:-1].index #get the dates from the nestle close price series
nestle_close_price_index=pd.to_datetime(nestle_close_price_index) #and convert to datetime objects
nestle_close_price_index=nestle_close_price_index.date            #drop the HH:MM:SS data and just keep the date.j
print(nestle_close_price_index)
print('**********************************************************')


price_18aug2021 = nestle_close_price.iloc[-420] # store the closing share price on the 18/08/2021 as a constant
print('The price on 18/08/2021 is: ', price_18aug2021)  #float64 value and print
 #variable lists have been set up to store share price simulation data
get_price=[price_18aug2021]+[0]*420 #they will be used to reset share prices between simulations
#Here 1400 different simulations will be run

for l in range(0,2000):    #1400 simulations will be run
   y_1 = price_18aug2021    #reset y_1 and y_2 to the close price on 18/aug/2021 before the next
   y_2 = price_18aug2021    #monte carlo run of prices is generated
   resamp_prices=get_price #reset resamp_prices array before it accepts 10 new resampled additions from the k indexed loop
   for k in range(0,420):
      y_2 = y_2+y_2*(np.random.choice(nestle_daily_returns)) #multiply the previous day's closing price by a historicallly resampled return value
      resamp_prices[k+1]=y_2
   resamp_series = pd.Series(resamp_prices)  #
   RESAMP = pd.concat([RESAMP, resamp_series], axis=1)  #
RESAMP_SIM_ONLY = RESAMP #Define two new data frames that emphasise the storage of data that is simulated not actual
print('**********************************************************')

print(RESAMP_SIM_ONLY.head())
print('**********************************************************')

MEDIAN_RS=RESAMP_SIM_ONLY.median(axis=1) #get the median value of each row and store in the series MEDIAN_RS
MEAN_RS =RESAMP_SIM_ONLY.mean(axis=1).tolist() #get the mean value of each row and store in the series MEAN_RS
STD_RS=RESAMP_SIM_ONLY.std(axis=1).tolist() #get the standard deviation of each row and store in the series STD_RS
MEAN_RS_array=np.array(MEAN_RS) #convert the series of means to an array
STD_RS_array=np.array(STD_RS) #convert the series of standard deviations to an array
#Calculate the 68% and 95% confidance intervals for the historically sampled simulated share price paths
MEAN_RS_plus_1STD=MEAN_RS_array+STD_RS_array;MEAN_RS_minus_1STD=MEAN_RS_array-STD_RS_array
MEAN_RS_plus_2STD=MEAN_RS_array+2*STD_RS_array;MEAN_RS_minus_2STD=MEAN_RS_array-2*STD_RS_array
First_RESAMP_SIM_ONLY_sim = RESAMP_SIM_ONLY.iloc[:, 0:1] #Extract the random walk from the first column as an example

#Main plotting section of code --------------------------------------------------------------------------------------
import matplotlib.pyplot as plt #Import the matplotlib.ptplot libary
              #Import numpy for numerical calculations
            #Import the pandas data open source data analysis and manipulation tool
import seaborn as sns           #Import the Seaborn libary for plotting
import datetime as datetime     #Import the datetime libary for display
from fitter import Fitter, get_common_distributions, get_distributions
#First plot the actual nestle daily close price path from 18/09/2021 for 10 days
fig,ax=plt.subplots()


ax.plot(NESTLE_CLOSE,label="Actual Close price",color='b',marker='.')
ax.legend()
ax.plot(MEAN_RS, color='g',label="MEAN")
ax.legend()
ax.plot(MEDIAN_RS,color='c',label="MEDIAN")
ax.legend()
ax.plot(MEAN_RS_plus_1STD, color='r',linestyle='--',label="MEAN+1sigma")
ax.legend()
ax.plot(MEAN_RS_minus_1STD, color='r',linestyle='--',label="MEAN-1sigma")
ax.legend()
ax.plot(MEAN_RS_plus_2STD,color='m',linestyle='--',label="MEAN+2sigma")
ax.legend()
ax.plot(MEAN_RS_minus_2STD,color='m',linestyle='--',label="MEAN-2sigma")
ax.legend()
ax.set_title('1400 sim runs of resampled data v actual close price  ')
ax.set_xlabel('Days since 18/08/2021')
ax.set_ylabel('Close Price')
plt.show()
#Plot a 3 x 3 grid of typical gaussian generated close price paths from 18/09/2021 +10 stock market days.

#------------------------

#Plot a 3 x 3 grid of typical historically random sampled close price paths from 18/09/2021 +10 stock market days.
fig,ax=plt.subplots(3,3)
ax[0][0].plot(NESTLE_CLOSE, color='b')
ax[0][0].plot(RESAMP_SIM_ONLY.iloc[:, 0:1], color='r')
ax[0][0].set_title('Historical sim v nestle close')
ax[0][0].set_xlabel('')
ax[0][0].set_ylabel('Close Price')
#---
ax[0][1].plot(NESTLE_CLOSE, color='b')
ax[0][1].plot(RESAMP_SIM_ONLY.iloc[:, 1:2], color='r')
ax[0][1].set_title('Historical sim v nestle close')
ax[0][1].set_xlabel('')
ax[0][1].set_ylabel('')
#---
ax[0][2].plot(NESTLE_CLOSE, color='b')
ax[0][2].plot(RESAMP_SIM_ONLY.iloc[:, 2:3], color='r')
ax[0][2].set_title('Historical sim v nestle close')
ax[0][2].set_xlabel('')
ax[0][2].set_ylabel('')
#---
ax[1][0].plot(NESTLE_CLOSE, color='b')
ax[1][0].plot(RESAMP_SIM_ONLY.iloc[:, 3:4], color='r')
ax[1][0].set_title('')
ax[1][0].set_xlabel('Days since 18/08/2021')
ax[1][0].set_ylabel('Close Price')

ax[1][1].plot(NESTLE_CLOSE, color='b')
ax[1][1].plot(RESAMP_SIM_ONLY.iloc[:, 4:5], color='r')
ax[1][1].set_title('')
ax[1][1].set_xlabel('')
ax[1][1].set_ylabel('')

ax[1][2].plot(NESTLE_CLOSE, color='b')
ax[1][2].plot(RESAMP_SIM_ONLY.iloc[:, 5:6], color='r')
ax[1][2].set_title('')
ax[1][2].set_xlabel('')
ax[1][2].set_ylabel('')

ax[2][0].plot(NESTLE_CLOSE, color='b')
ax[2][0].plot(RESAMP_SIM_ONLY.iloc[:, 6:7], color='r')
ax[2][0].set_title('')
ax[2][0].set_xlabel('Days since 18/08/2021')
ax[2][0].set_ylabel('Close Price')

ax[2][1].plot(NESTLE_CLOSE, color='b')
ax[2][1].plot(RESAMP_SIM_ONLY.iloc[:, 7:8], color='r')
ax[2][1].set_title('')
ax[2][1].set_xlabel('Days since 18/08/2021')
ax[2][1].set_ylabel('')

ax[2][2].plot(NESTLE_CLOSE, color='b')
ax[2][2].plot(RESAMP_SIM_ONLY.iloc[:, 8:9], color='r')
ax[2][2].set_title('')
ax[2][2].set_xlabel('Days since 18/08/2021')
ax[2][2].set_ylabel('')
#------------------------

p=sns.displot(nestle_daily_returns, kind='hist', bins = 100, aspect = 1.5)
p.set(title = "Title")

f = Fitter(nestle_daily_returns,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm",
                          "t"])
f.fit()
print(f.summary())
fig,ax=plt.subplots()
nestle_daily_returns.plot(kind='hist',bins=50)
plt.xlabel('Daily returns')
plt.ylabel('Count')
plt.title('Histogram of actual daily returns of nestle share price from 04/10/2010-18/08/2021')


print(RESAMP_SIM_ONLY.iloc[:,0:5].head(5))
