# 제 5회 기출 복원 2유형 연습

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('./workpython/carprice.csv')
df.info()

"""
문제
차량 가격을 예측
"""
y = df['price']

#print(df['model'].unique())
#print(df['fuelType'].unique())
"""
[' A1' ' A6' ' A4' ' A3' ' Q3' ' Q5' ' A5' ' S4' ' Q2' ' A7' ' TT' ' Q7'
 ' RS6' ' RS3' ' A8' ' Q8' ' RS4' ' RS5' ' R8' ' SQ5' ' S8' ' SQ7' ' S3'
 ' S5' ' A2' ' RS7']
['Petrol' 'Diesel' 'Hybrid']
"""

newdf = df.copy()
newdf['model'] = newdf['model'].astype('category')
newdf['fuelType'] = newdf['fuelType'].astype('category')
newdf['transmission'] = newdf['transmission'].astype('category')

encoder = LabelEncoder()

newdf['model'] = encoder.fit_transform(newdf['model'])
newdf['fuelType'] = encoder.fit_transform(newdf['fuelType'])
newdf['transmission'] = encoder.fit_transform(newdf['transmission'])

#print(newdf['model'].unique())
#print(newdf['fuelType'].unique())
"""
[ 0  5  3  2  9 10  4 20  8  6 25 11 17 14  7 12 15 16 13 23 22 24 19 21 1 18]
[2 0 1]
"""
corr_table = newdf[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax'
                    ,'mpg' ,'engineSize', 'price']].corr()

print(corr_table['price'])
"""
model           0.394635
year            0.592581
transmission    0.009864
mileage        -0.535357
fuelType       -0.032135
tax             0.356157
mpg            -0.600334
engineSize      0.591262
price           1.000000
"""



