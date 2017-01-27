import pandas as pd
import numpy as np

# Format data by using Panda
# Each row will have our input (3 days) and output (prediction of the following day)
# Note: header of data file should be (modified):
# Site,Parameter,Date (LST),Year,Month,Day,Hour,Value,Unit,FooUnit,Duration,QC_Name

NUMBER_OF_DAYS = 4
df = pd.read_csv("test-data.csv", sep=',|\s', engine='python')
df.loc[df['QC_Name'] != 'Valid', 'Value'] = 0 #invalid record -> 0
df['Date'] = pd.to_datetime(df['Date'] )      #otherwise row order get messed up
df_daily = df.pivot_table(index='Date', columns='Hour', values='Value')
df_daily.reset_index(drop=True, inplace=True)
df_daily['Partition'] = 1+df_daily.index//NUMBER_OF_DAYS
df_daily['Columns'] = (df_daily.index%NUMBER_OF_DAYS)
df_n_days = df_daily.pivot_table(index='Partition', columns='Columns')
sorted_column_head = sorted(df_n_days.columns, key=lambda x: x[1])
df_n_days = df_n_days[sorted_column_head]
# Not necessary to write to csv file
df_n_days.to_csv('formatted.csv', sep=',')
print("Completed formatting!")
