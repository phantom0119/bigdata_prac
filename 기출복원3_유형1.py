
import pandas as pd
import numpy as np


df = pd.read_csv('./workpython/country.csv')
print(df.head())
df.info()
newdf = df.dropna()

newdf.info()

gdf = newdf['Guam'].quantile(0.3)
print(gdf)

#------------------------------------------------------

t2000 = newdf[newdf['year'] == 2000]
print(t2000)

cnt = 0
for t in t2000.drop(['year'], axis=1) :
    print(t2000[t].values)
    if t2000[t].values > 119.7 :
        cnt +=1

print(cnt)

#-----------------------------------------------------

dfnew = t2000.iloc[:, 1:8]  # 'yaer' Column 제외
print(dfnew[dfnew>119.7].count().sum())


#-----------------------------------------------------

print(df.isnull().sum())