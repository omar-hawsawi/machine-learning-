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
df = df.rename(columns={20: 'Date', 5: 'Amount', 2:'Country',3:'Product',4:'Sales Person'})

df['Date'] = pd.to_datetime(df['Date']) 

df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True)
df['Amount'] = pd.to_numeric(df['Amount'])

print(df.dtypes)

print(df.describe(include='all'))
plt.figure(figsize=(8,6))
sns.histplot(df[6:100 ], bins=20) 
plt.title("Distribution of Boxes Shipped")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df[1:100], bins=20)
plt.title("Distribution of Revenue")
plt.show()

country_revenue = df.groupby('Country')['Amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(10,5))
country_revenue.plot(kind='bar')
plt.title("Total Revenue by Country")
plt.ylabel("Revenue")
plt.show()

print(country_revenue)

product_revenue = df.groupby('Product')['Amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(10,5))
product_revenue.plot(kind='bar')
plt.title("Revenue by Product")
plt.ylabel("Revenue")
plt.show()

print(product_revenue)

salesperson_revenue = (
    df.groupby('Sales Person')['Amount'].sum().sort_values(ascending=False)
)

print(salesperson_revenue.head(10))

plt.figure(figsize=(8,5))
sns.scatterplot(x='Date', y='Amount', data=df)
plt.title("Date vs Revenue")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df[['Date', 'Amount']].corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

df['Month'] = df['Date'].dt.to_period('M')

monthly_revenue = df.groupby('Month')['Amount'].sum()

plt.figure(figsize=(10,5))
monthly_revenue.plot()
plt.title("Monthly Revenue Trend")
plt.ylabel("Revenue")
plt.show()