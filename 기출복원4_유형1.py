

import pandas as pd
import numpy as np

df = pd.read_csv('./workpython/women.csv')
print(df.head())

w3 = df['weight'].quantile(0.75)
w1 = df['weight'].quantile(0.25)
value = abs(w3-w1)

print(int(value))


#----------------------------------------------------

df = pd.read_csv('./workpython/USvideos.csv')
print(df.head())

df['ratio'] = df['likes'] / df['views']

comp = (df['ratio'] > 0.04) & (df['ratio'] < 0.05)
newdf = df[(df['category_id'] == 10) & (comp)]

print(len(newdf))