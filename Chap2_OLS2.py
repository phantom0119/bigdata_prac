# statsmodel.formula.api의 ols 모듈로 다중 회귀 모델 실습

import pandas as pd
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

df = pd.read_csv('workpython/seatbelts.csv', index_col=0)
df.info()
"""
Index: 192 entries, 1 to 192
Data columns (total 8 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   DriversKilled  192 non-null    int64  
 1   drivers        192 non-null    int64  
 2   front          192 non-null    int64  
 3   rear           192 non-null    int64  
 4   kms            192 non-null    int64  
 5   PetrolPrice    192 non-null    float64
 6   VanKilled      192 non-null    int64  
 7   law            192 non-null    int64  
dtypes: float64(1), int64(7)
memory usage: 13.5 KB
"""

# print(df.iloc[:, 0:5].head())
# print(df.iloc[:, 5:8].head())
"""
    drivers  front  rear    kms   PetrolPrice  VanKilled  law   DriversKilled
1     1687    867   269   9059      0.102972         12    0            107 
2     1508    825   265   7685      0.102363          6    0            97
3     1507    806   319   9963      0.102062         12    0            102
4     1385    814   407  10955      0.100873          8    0            87
5     1632    991   454  11823      0.101020         10    0            119
"""

# DriversKilled를 종속변수로 하고, 상관 계수를 통해 유의미한 관계를 분석하기.
cor = df.corr()
#print(abs(cor['DriversKilled']))
"""
DriversKilled    1.000000
drivers          0.888826
front            0.706760
rear             0.353351
kms              0.321102
PetrolPrice      0.386606
VanKilled        0.407041
law              0.328505
Name: DriversKilled, dtype: float64

-> drivers, front, VanKilled, PetrolPrice 값이 상위 4개의 유의 관계를 보여준다.
"""

dfx = df[['drivers','front','VanKilled','PetrolPrice']]     # 독립변수
y = df[['DriversKilled']]                                   # 종속변수

#print(dfx.head())
#print(y.head())



# 독립변수의 경우 모두 연속형 자료이므로 학습 데이터로 그대로 사용할 수 있다.
# 하지만 데이터 값의 크기에 영향을 받기 때문에 일반적으로 스케일링(Z-score, Min-Max)을 적용하는 것이 좋다.
scaler =  StandardScaler()
scaled = scaler.fit_transform(dfx)
scaled_x = pd.DataFrame(scaled, columns=['drivers','front','VanKilled','PetrolPrice'])
scaled_x.index = range(1, len(scaled_x)+1)

#print(scaled_x)
"""
      drivers     front  VanKilled  PetrolPrice
0    0.057789  0.170527   0.811240    -0.053705
1   -0.561897 -0.069964  -0.842828    -0.103837
2   -0.565359 -0.178758   0.811240    -0.128582
3   -0.987715 -0.132950  -0.291472    -0.226506
4   -0.132617  0.880549   0.259884    -0.214453
..        ...       ...        ...          ...
187 -1.337371 -1.106368  -1.118506     0.920035
188 -0.783461 -1.112094  -0.567150     0.862078
189 -0.329948 -1.123546  -0.567150     1.057430
190  0.230886 -0.722727  -1.394184     1.021247
191  0.320896 -0.665467  -0.567150     1.024591
[192 rows x 4 columns]
"""

#print(y)


newdf = pd.concat([scaled_x, y], axis=1)
print(newdf.head())



# trainx, trainy, testx, testy = train_test_split(dfx, y, test_size=0.2, random_state=2024)

model = ols('y ~ scaled_x', data=df).fit()

fitted = model.fittedvalues

#print(type(fitted))  # <class 'pandas.core.series.Series'>

print(model.summary())






