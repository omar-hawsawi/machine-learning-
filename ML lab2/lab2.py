import pandas as pd
#task 1
df=pd.read_csv("spambase/spambase.data", header=None)
print("dataset type")
print(df.dtypes,"\n")
print("\nColumn names:", df.columns.tolist())

print("dataset shape")
print(df.shape)

print("first 5 rows of the dataset")
print(df.head())
