#lab 3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = pd.read_csv("spambase/spambase.data", header=None)
sns.set()
print(df.head())
print(df.isna())
print(df.isna().sum())
df.duplicated()[df.duplicated()==True]
print("Shape (rows, columns): ", df.shape,"\n")

print("number of rows: ", df.shape[0])

print("number of columns: ", df.shape[1])

print(df.dtypes)


print ("")
print("after the changes \n")
df = df.rename(columns={0: 'Date', 5: 'Amount'})

df['Date'] = pd.to_datetime(df['Date']) 
df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
df['Amount'] = pd.to_numeric(df['Amount'])

print(df.dtypes)

print(df.describe(include='all'))