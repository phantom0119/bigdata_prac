
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./workpython/Boston.csv')
newdf = df.sort_values(['crim'], ascending=False, inplace=False)
newdf.reset_index(drop=True, inplace=True)

print(newdf['crim'].iloc[0:10])

rev = newdf.iloc[9].crim
print(rev)
newdf['crim'].iloc[0:10] = rev


q3 = newdf[newdf['age'] >= 80]
print(q3.crim.mean())


#-------------------------------------------------------------------------

df = pd.read_csv('./workpython/housing.csv')

r = int(len(df)*0.8)  #16512.0

data1 = df.iloc[0:r]

t1 = data1['total_bedrooms'].dropna()
t2 = data1['total_bedrooms'].fillna(data1['total_bedrooms'].median())

print(t1.std(), t2.std())


#------------------------------------------------------------------------

df = pd.read_csv('./workpython/insurance.csv')
print(df.head())
m = df['charges'].mean()
n = df['charges'].std()
outlier = m + 1.5*n

tmp = df[df['charges'] >= outlier]
tmps = tmp['charges'].sum()
print(m, n, outlier, tmps)
