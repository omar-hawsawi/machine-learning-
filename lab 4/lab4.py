import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

sns.set(style="whitegrid")
df=pd.read_csv('spambase.data',header=None)
print(df.head())

print(df.isna().sum())


df_missing = df.copy()
df_missing.loc[0:5, 10] = np.nan
print(df_missing.isna().sum())
print(df.shape)
print(df_missing.shape)

print(df_missing.head(10))

df_removed = df_missing.dropna()
print(df_removed.shape)

print(df_removed.isna().sum())

df_imputed_mean = df_missing.copy()
df_imputed_mean[10] = df_imputed_mean[10].fillna(df_imputed_mean[10].mean())

print(df_imputed_mean.isna().sum())

print(df_imputed_mean.head(10))

df_imputed_median = df_missing.copy()
df_imputed_median[10] = df_imputed_median[10].fillna(df_imputed_median[10].median())

df_imputed_median.isna().sum()

df_imputed_median = df_missing.copy()
print(df_imputed_median.isna().sum())

print(df_imputed_median.head(10))
plt.figure(figsize=(6,4))
sns.boxplot(x=df[10])
plt.title("Boxplot of Revenue")
plt.show()


Q1 = df[10].quantile(0.25)
Q3 = df[10].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df[10] < lower) | (df[10] > upper)]
print(outliers.head(15))


df_no_outliers = df[(df[10] >= lower) & (df[10] <= upper)]
print("Original shape: ",df.shape)
print("After removing outliers: ",df_no_outliers.shape)


lower_cap = df[10].quantile(0.05)
upper_cap = df[10].quantile(0.95)

df_capped = df.copy()
df_capped[10] = df_capped[10].clip(lower_cap, upper_cap)
print(df_capped[10].describe())

print( df[[10, 20]].head())

scaler = StandardScaler()
df_standardized = df[[10, 20]].copy()

df_standardized[[10,20]] = scaler.fit_transform(df_standardized)

print(df_standardized.head())


plt.figure(figsize=(6,4))
sns.heatmap(df_standardized[[10,20]].corr(), 
            annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Before PCA)")
plt.show()

X = df_standardized[[10, 20]]

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(6,4))
plt.scatter(principal_components[:,0], principal_components[:,1])
plt.title("PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()