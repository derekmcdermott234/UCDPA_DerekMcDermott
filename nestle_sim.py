#This is a python script to try and predict the change in the nestle
#share price on the Swiss stock exchange.Here I am attempting to predict
#the nestle share price 10 days into the future based on the past 10years
#of share price data for nestle as obtained fron Yahoo Finance.
import matplotlib.pyplot as plt #Import the matplotlib.ptplot libary
import numpy as np              #Import numpy for numerical calculations
import pandas as pd             #Import the pandas data open source data analysis and manipulation tool
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
nestle=pd.read_csv('C:/Users/derek/Desktop/ucd/Working Data/'   #I am importing data from Yahoo for the nestle share price
                  'yahoo data/NESN.SW.csv',                     #from 14/10/2010-13/09/2021
                   parse_dates=['Date'],index_col='Date')       #The data is imported as a csv file and read into the dataframe
                                                                #nestle,the dates in the date column are converted to datetime object
                                                                #The date column is set as the index column.

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
nestle_close_price=nestle_clean['Close']            #copy the Close price column from nestle_clean to a new
def check_for_zeros(x):               #define function to check for zeros in a series or list
   nestle_int_check=x.astype(int)
   return nestle_int_check[nestle_int_check==0].count()
print(check_for_zeros(nestle_close_price))     #call on the function to check nestle_close_price data for zeros
print(nestle_close_price) #print the nestle close price data to inspect that the first date
                          # and last date are as expected

nestle_daily_returns=nestle_close_price.pct_change().dropna() #calculate the day by day returns on the daily
                                                              #closing price data and drop the first value which is NaN
mu = nestle_daily_returns.loc[:'2021-08-18'].mean() #calcullate the mean of the daily returns up to 18/08/2021
standard_dev=nestle_daily_returns.loc[:'2021-08-18'].std() #calcullate the standard deviation of the daily returns up to 17/08/2021
print('...')
print(mu,standard_dev) #print the mean and standard deviation of the daily returns series
print('...')

Index_1=[0,1,2,3,4,5,6,7,8,9] #create an index for a new dataframe with 10 rows
neww=pd.DataFrame() #A new dataframe is created to store the simulated stock price data
neww['Index']=Index_1 #assign the values in the Index_1 list to the neww dataframe column 'Index'

neww=neww.set_index('Index') #set the index of the new dataframe to Index_1 i.e.0 to 9
neww['Actual Nestle Close Price'] = np.array(nestle_close_price.iloc[2729:2739]) #10 nestle data points for the actual closing
                                                                                 # price from 18/08/2021-31/08/2021


print(neww['Actual Nestle Close Price'])
phi= np.random.normal(loc=mu, scale=standard_dev, size=(1,10000)) #create a 1 x 1000 array of random numbers
                                                                 #drawn from a gussian distribution with the
                                                                 #mean and standard deviation of all daily returns
price_18aug2021 = nestle_close_price.loc['18/08/2021']  # store the closing share price on the 18/08/2021 as a constant
print('The price on 18/08/2021 is: ', price_18aug2021)
set_price=[price_18aug2021,0,0,0,0,0,0,0,0,0] #create a constant list of 20 values to be referenced
                                                                  #within the loop structure to follow.#in one run generate 20 simulated closing prices from 18/08/2021 onwards and add this singe run as a column of data
#to the right of the neww dataframe
#the columns with 20 data entries are added to the neww dataframe 100 times,100 runs of 20 days
for l in range(0,100):    #100 simulations will be run
   y = price_18aug2021    #reset y to the close price on 18/aug/2021
   prices_1=set_price     #reset prices_1 array before it accepts new values from the k indexed loop
   for k in range(0,9):
      y = y+y*(phi[0][l+k]) #multiply the previous day's closing price by a randomly generated return value
      prices_1[k+1]=y       #store the next day's simulated closing price in a list
   d_series=pd.Series(prices_1) #convert the list to a series before you can add it to neww
   neww=pd.concat([neww,d_series],axis=1) #add a column to the neww dataframe that contains 1 run of
                                          # 20 simulated closing prices
c_neww=neww.drop("Actual Nestle Close Price",axis=1) #create a new dataframe with the actual nestle closing
                                                     # prices column dropped
#the average run from 100 simulated runs of 20 days is computed
c_neww['MEAN']=c_neww.mean(axis=1) #get the mean value of each row and store in a new column 'MEAN'
print(c_neww['MEAN'].head())
print(c_neww.info())
print('...')
print(neww.info())
print(neww)
print('...')
sim_2 = neww.iloc[:, 1:2];
sim_3 = neww.iloc[:, 2:3];
sim_4 = neww.iloc[:, 3:4];
sim_5 = neww.iloc[:, 4:5]
sim_6 = neww.iloc[:, 5:6];
sim_7 = neww.iloc[:, 6:7];
sim_8 = neww.iloc[:, 7:8];
sim_9 = neww.iloc[:, 8:9]
fig,ax=plt.subplots(1,2) #defines figure/fig which is the canvas that may contain 1 or more Axes
                         #defines Axes/ax represents an individual plot drawn on figure
plt.xlabel('x')
plt.ylabel('y')
ax[0].plot(neww['Actual Nestle Close Price'], color='b')
ax[0].plot(c_neww['MEAN'], color='r')
ax[0].set_title('Average sim run v actual ')
ax[0].set_xlabel('Days since 18/08/2021')
ax[0].set_ylabel('Close Price')
ax[1].plot(neww['Actual Nestle Close Price'], color='b')
ax[1].plot(sim_2, color='r')
ax[1].set_title('Single sim run v actual ')
ax[1].set_xlabel('Days since 18/08/2021')
ax[1].set_ylabel('Close Price')


#nestle_daily_returns.plot(kind='hist')
#sns.set_style('white')
#sns.set_context("paper", font_scale = 2)
sns.displot(nestle_daily_returns, kind='hist', bins = 100, aspect = 1.5).set(title='Title of Plot')
plt.title('Life Expectancy')
plt.xlabel('Life Exp (years)')
plt.ylabel('Frequency')
f = Fitter(nestle_daily_returns,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm",
                          "t"])
f.fit()
print(f.summary())
plt.show()#

#the averaged run data from 100 simulated runs of 20 days is stored in a csv file
#c_neww['MEAN'].to_csv('C:/Users/derek/Desktop/ucd/Working Data/yahoo data/nestle_mean_columns.csv',mode='a', index=False, header=True)

