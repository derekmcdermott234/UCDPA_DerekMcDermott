import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as datetime

df_1=pd.read_csv('C:/Users/derek/Desktop/ucd/Working Data/'   #I am importing data from Yahoo for the nestle share price
                  'yahoo data/500_Person_Gender_Height_Weight_Index.csv')                    #from 14/10/2010-13/09/2021
df_na=df_1.dropna()
df_na_dropped=df_na.drop_duplicates()
print(df_1.shape,df_na.shape,df_na_dropped.shape)
df=df_na_dropped
print(df.isna().count())

print(df.head())
grouped_H=df.groupby("Gender")["Height"].mean()
grouped_W=df.groupby("Gender")["Weight"].mean()
grouped_BMI=df.groupby("Gender")["Index"].mean()
print(grouped_H.head())
print(grouped_W.head())
print(grouped_BMI.head())
MIN_MAX_SUM_H=df.groupby("Gender")["Height"].agg([min,max,sum])
print(MIN_MAX_SUM_H)
grouped_Gender_Weight_Height=df.groupby("Gender")[["Weight","Height","Index"]].mean()
print(grouped_Gender_Weight_Height)
df_pivot_table_W=df.pivot_table(values="Weight",index="Gender")
print(df_pivot_table_W)
df_pivot_table_H=df.pivot_table(values="Height",index="Gender")
print(df_pivot_table_H)
median_summary_stat=df.pivot_table(values="Height",index="Gender",aggfunc=np.median)
print(median_summary_stat)