import pandas as pd
import numpy as np
import glob
import os

# Format data by using Panda
# Each row will have output (24), Month, Day, and  our input
# Note: header of data file should be (modified):
# Site,Parameter,Date (LST),Year,Month,Day,Hour,Value,Unit,FooUnit,Duration,QC_Name

NUMBER_OF_DAYS = 10
df = pd.read_csv("not-formatted-data.csv", sep=',|\s', engine='python')
# df = pd.read_csv("test-data.csv", sep=',|\s', engine='python')

df.loc[df['QC_Name'] != 'Valid', 'Value'] = 0 #invalid record -> 0
df['Date'] = pd.to_datetime(df['Date'] )      #otherwise row order get messed up
df_daily = df.pivot_table(index='Date', columns='Hour', values='Value')
# df_daily['Month'] = pd.DatetimeIndex(df_daily.index).month
# df_daily['Day'] = pd.DatetimeIndex(df_daily.index).day
dfs = []
for i in reversed(range(NUMBER_OF_DAYS)):
    temp_df = df_daily.ix[i:]
    # if i != NUMBER_OF_DAYS-1:
    #     del temp_df['Month']
    #     del temp_df['Day']
    temp_df.reset_index(drop=True, inplace=True)
    dfs.append(temp_df)
df_inclusive = pd.concat(dfs, axis=1)
# print(df_inclusive)
df_inclusive.to_csv('sliding-days-10.csv',header=False, sep=',')
print("Completed formatting!")
