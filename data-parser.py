import pandas as pd
import numpy as np
import glob
import os

# Format data by using Panda
# Each row will have our input (3 days) and output (prediction of the following day)
# Note: header of data file should be (modified):
# Site,Parameter,Date (LST),Year,Month,Day,Hour,Value,Unit,FooUnit,Duration,QC_Name

NUMBER_OF_DAYS = 5
#Load all data files
# path =os.getcwd() +'/all-data' # use your path
# allFiles = glob.glob(path + "/*.csv")
# df = pd.DataFrame()
# list_ = []
# for file_ in allFiles:
#     frame = pd.read_csv(file_,sep=',|\s',index_col=None, engine='python')
#     list_.append(frame)
# df = pd.concat(list_)

df = pd.read_csv("not-formatted-data.csv", sep=',|\s', engine='python')
# df = pd.read_csv("test-data.csv", sep=',|\s', engine='python')

df.loc[df['QC_Name'] != 'Valid', 'Value'] = 0 #invalid record -> 0
df['Date'] = pd.to_datetime(df['Date'] )      #otherwise row order get messed up
df_daily = df.pivot_table(index='Date', columns='Hour', values='Value')
dfs = []
# print(df_daily)
for i in range(NUMBER_OF_DAYS):
    sliding_df = df_daily.ix[i:]
    sliding_df.reset_index(drop=True, inplace=True)
    sliding_df['Partition'] = 1+sliding_df.index//NUMBER_OF_DAYS
    sliding_df['Columns'] = (sliding_df.index%NUMBER_OF_DAYS)
    sliding_df = sliding_df.pivot_table(index='Partition', columns='Columns')
    dfs.append(sliding_df)
df_inclusive = pd.concat(dfs)
# print(df_inclusive)
sorted_column_head = sorted(df_inclusive.columns, key=lambda x: x[1])
df_inclusive = df_inclusive[sorted_column_head]
# # Not necessary to write to csv file
df_inclusive.to_csv('sliding-days-data-5.csv',header=False, sep=',')

print("Completed formatting!")
