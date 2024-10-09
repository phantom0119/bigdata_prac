"""
Kaggle Dataset 적용

문제: test Dataset에 포함된 각 시간 동안 임대된 자전거의 총 수를 예측.
  - 종속변수는 'count' Column.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgbm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

traindf = pd.read_csv('workpython/bike_sharing_train.csv')
testdf = pd.read_csv('workpython/bike_sharing_test.csv')

#traindf.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   datetime    10886 non-null  object 
 1   season      10886 non-null  int64  
 2   holiday     10886 non-null  int64  
 3   workingday  10886 non-null  int64  
 4   weather     10886 non-null  int64  
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64  
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64  
 10  registered  10886 non-null  int64  
 11  count       10886 non-null  int64  
dtypes: float64(3), int64(8), object(1)
memory usage: 1020.7+ KB
"""

#testdf.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6493 entries, 0 to 6492
Data columns (total 9 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   datetime    6493 non-null   object 
 1   season      6493 non-null   int64  
 2   holiday     6493 non-null   int64  
 3   workingday  6493 non-null   int64  
 4   weather     6493 non-null   int64  
 5   temp        6493 non-null   float64
 6   atemp       6493 non-null   float64
 7   humidity    6493 non-null   int64  
 8   windspeed   6493 non-null   float64
dtypes: float64(3), int64(5), object(1)
memory usage: 456.7+ KB
"""

#print(traindf.head())
"""
              datetime  season  holiday  ...  casual  registered  count
0  2011-01-01 00:00:00       1        0  ...       3          13     16
1  2011-01-01 01:00:00       1        0  ...       8          32     40
2  2011-01-01 02:00:00       1        0  ...       5          27     32
3  2011-01-01 03:00:00       1        0  ...       3          10     13
4  2011-01-01 04:00:00       1        0  ...       0           1      1
"""

# 중간 점검
# train과 test DataFrame에서 결측값(None, NaN)은 확인되지 않는다.
# train과 test DataFrame의 Column 구성이 다르다.  따라서 train에서 test에 맞게 컬럼 일부를 삭제해야 한다.
# Data 설명(Dataset Description)에서 'season', 'holiday', 'workingday', 'weather' Column은 정수로 작성된 "범주형" 데이터이다.
#  --> 정확한 의미를 위해 category 범주화를 수행해야 한다.


# test에는 없는 'casual', 'registered' column 제거
traindf = traindf.drop(columns=['casual', 'registered'], axis=1)

# 'season', 'holiday', 'workingday', 'weather' Column 확인.
print(traindf.season.unique())      #  1 = spring, 2 = summer, 3 = fall, 4 = winter
print(traindf.holiday.unique())     #  whether the day is considered a holiday
print(traindf.workingday.unique())  # whether the day is neither a weekend nor holiday
print(traindf.weather.unique())     # 1: Clear, Few clouds, Partly cloudy, Partly cloudy
                                    # 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
                                    # 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
                                    # 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

# 위 4개의 Column을 범주화하기
traindf['season'] = traindf['season'].astype('category')
traindf['holiday'] = traindf['holiday'].astype('category')
traindf['workingday'] = traindf['workingday'].astype('category')
traindf['weather'] = traindf['weather'].astype('category')

testdf['season'] = testdf['season'].astype('category')
testdf['holiday'] = testdf['holiday'].astype('category')
testdf['workingday'] = testdf['workingday'].astype('category')
testdf['weather'] = testdf['weather'].astype('category')

#traindf.info()
"""
Index: 10886 entries, 2011-01-01 00:00:00 to 2012-12-19 23:00:00
Data columns (total 9 columns):
 #   Column      Non-Null Count  Dtype   
---  ------      --------------  -----   
 0   season      10886 non-null  category
 1   holiday     10886 non-null  category
 2   workingday  10886 non-null  category
 3   weather     10886 non-null  category
 4   temp        10886 non-null  float64 
 5   atemp       10886 non-null  float64 
 6   humidity    10886 non-null  int64   
 7   windspeed   10886 non-null  float64 
 8   count       10886 non-null  int64   
dtypes: category(4), float64(3), int64(2)
memory usage: 553.4+ KB
"""

# 'datetime' Column을 Datetime 자료로 바꾸고 분석 목적의 '연', '월', '일', '시간' Column으로 나누기.
# '일'의 경우 날짜 구분보다 '월~일' 구분이 holiday, workingday와 연계할 수 있는 정도가 좋으므로 weekday 범주로 변환한다.
traindf['datetime'] = pd.to_datetime(traindf['datetime'])

traindf['year'] = traindf['datetime'].dt.year
traindf['month'] = traindf['datetime'].dt.month
traindf['weekday'] = traindf['datetime'].dt.day_name()
traindf['hour'] = traindf['datetime'].dt.hour

traindf['year'] = traindf['year'].astype('category')
traindf['month'] = traindf['month'].astype('category')
traindf['weekday'] = traindf['weekday'].astype('category')
traindf['hour'] = traindf['hour'].astype('category')


# 'datetime' Column의 역할은 다 했으니 버리기
traindf = traindf.drop(columns=['datetime'], axis=1)

#print(traindf.head())
"""
   season holiday workingday weather  temp  ...  count  year  month   weekday  hour
0      1       0          0       1  9.84  ...     16  2011      1  Saturday     0
1      1       0          0       1  9.02  ...     40  2011      1  Saturday     1
2      1       0          0       1  9.02  ...     32  2011      1  Saturday     2
3      1       0          0       1  9.84  ...     13  2011      1  Saturday     3
4      1       0          0       1  9.84  ...      1  2011      1  Saturday     4

[5 rows x 13 columns]
"""

#testdf도 같은 구조로 만들기
testdf['datetime'] = pd.to_datetime(testdf['datetime'])

testdf['year'] = testdf['datetime'].dt.year
testdf['month'] = testdf['datetime'].dt.month
testdf['weekday'] = testdf['datetime'].dt.day_name()
testdf['hour'] = testdf['datetime'].dt.hour

testdf['year'] = testdf['year'].astype('category')
testdf['month'] = testdf['month'].astype('category')
testdf['weekday'] = testdf['weekday'].astype('category')
testdf['hour'] = testdf['hour'].astype('category')

testdf = testdf.drop(columns=['datetime'], axis=1)

#print(testdf.head())
"""
   season  holiday  workingday  weather  ...  year  month   weekday  hour
0       1        0           1        1  ...  2011      1  Thursday     0
1       1        0           1        1  ...  2011      1  Thursday     1
2       1        0           1        1  ...  2011      1  Thursday     2
3       1        0           1        1  ...  2011      1  Thursday     3
4       1        0           1        1  ...  2011      1  Thursday     4

[5 rows x 12 columns]
"""

# traindf의 'count' Column은 종속변수로 사용하므로 별도의 y로 옮기기
y = traindf['count']
traindf = traindf.drop(columns=['count'], axis=1)



# 'object', 'category'를 제외한 데이터 타입에 대해 StandardScaler 표준화 적용.
scaler_col = traindf.select_dtypes(exclude=['object', 'category']).columns

scaler = StandardScaler()
traindf[scaler_col] = scaler.fit_transform(traindf[scaler_col])
testdf[scaler_col] = scaler.transform(testdf[scaler_col])


#print(traindf[scaler_col])
"""
           temp     atemp  humidity  windspeed
0     -1.333661 -1.092737  0.993213  -1.567754
1     -1.438907 -1.182421  0.941249  -1.567754
2     -1.438907 -1.182421  0.941249  -1.567754
3     -1.333661 -1.092737  0.681430  -1.567754
4     -1.333661 -1.092737  0.681430  -1.567754
...         ...       ...       ...        ...
10881 -0.596935 -0.467310 -0.617666   1.617227
10882 -0.702182 -0.735182 -0.253919   0.269704
10883 -0.807428 -0.913959 -0.046064   0.269704
10884 -0.807428 -0.735182 -0.046064  -0.832442
10885 -0.912675 -0.824865  0.213755  -0.465608
[10886 rows x 4 columns]
"""

trainx, testx, trainy, testy = train_test_split(traindf, y, random_state=2024, test_size=0.2)


# 랜덤포레스트 모델은 모든 데이터 항목을 숫자 형태의 입력으로 받기 때문에 '범주형 문자열'인 경우는 인코딩을 통해 변환해야 한다.
# --> pd.get_dummies()를 이용한 One-Hot Encoding.
# --> LabelEncoder를 사용.
label_encoder = LabelEncoder()
trainx['weekday'] = label_encoder.fit_transform(trainx['weekday'])
testx['weekday'] = label_encoder.transform(testx['weekday'])


# 랜덤포레스트 예측 모형 적용
"""
랜덤포레스트에 적용할 수 있는 파라미터
# GridSearchCV를 활용한 RandomForest 최적화
grid_params = {
    'n_estimators' : [100, 150, 200, 250],
    'max_depth' : [None, 10, 20],
    'min_samples_split' : [2, 3, 4],
    'min_samples_leaf' : [1, 2, 3],
    'criterion' : ['gini', 'entropy'],
    'max_features' : ['sqrt', 'log2', 1, 2, 3, 0.5],
    'bootstrap' : [True, False],
    'class_weight' : [None, 'balanced']
}
"""
model = RandomForestRegressor(random_state=2024)
model.fit(trainx, trainy)
pred = model.predict(testx)
score = r2_score(testy, pred)

print(score)



# LightGBM 예측 모델 적용
# lgbmodel = lgbm.LGBMRegressor(random_state=2024, learning_rate=0.3)
# lgbmodel.fit(trainx, trainy)
#
# lgbm_pred = lgbmodel.predict(testx)
# mse = mean_squared_error(testy, lgbm_pred)
# r2 = r2_score(testy, lgbm_pred)
#
# print(mse, r2)



# 선형 회귀 모델 적용
# linmodel = LinearRegression()
# linmodel.fit(trainx, trainy)
#
# lin_pred = linmodel.predict(testx)
# mse = mean_squared_error(testy, lin_pred)
# r2 = r2_score(testy, lin_pred)
#
# print(mse, r2)