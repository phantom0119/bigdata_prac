# 회귀 모델 성능 평가 지표 연습
"""
실습 데이터 Boston.csv
- CRIM: 지역별 1인당 범죄율
- ZN: 25,000 평방피트당 주거용 토지 비율
- INDUS: 비소매상업지역 면적 비율
- CHAS: 찰스 강 인접 여부 (1: 강 인접, 0: 강 미인접)
- NOX: 일산화질소 농도 RM: 주택당 평균 방 개수
- AGE: 1940년 이전에 건축된 주택의 비율
- DIS: 5개의 보스턴 고용 센터와의 거리에 대한 가중치
- RAD: 방사형 고속도로 접근성 지수
- TAX: $10,000당 재산세율
- PTRATIO: 학생-교사 비율
- B: 1000(Bk - 0.63)^2, 여기서 Bk는 지역별 흑인 비율
- LSTAT: 저소득 계층의 비율
- MEDV: 주택 가격의 중앙값  (종속변수)
출처: https://kmrho1103.tistory.com/entry/머신러닝-보스턴-하우징-데이터 [데이터마이너를 꿈꾸며:티스토리]
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


df= pd.read_csv('./workpython/Boston.csv', encoding='euc-kr',
                index_col=0)

df.info()  # 결측값은 없는 것으로 확인
"""
<class 'pandas.core.frame.DataFrame'>
Index: 506 entries, 1 to 506
Data columns (total 14 columns):
 #   Column   Non-Null Count  Dtype  
---  ------   --------------  -----  
 0   crim     506 non-null    float64
 1   zn       506 non-null    float64
 2   indus    506 non-null    float64
 3   chas     506 non-null    int64  
 4   nox      506 non-null    float64
 5   rm       506 non-null    float64
 6   age      506 non-null    float64
 7   dis      506 non-null    float64
 8   rad      506 non-null    int64  
 9   tax      506 non-null    int64  
 10  ptratio  506 non-null    float64
 11  black    506 non-null    float64
 12  lstat    506 non-null    float64
 13  medv     506 non-null    float64
dtypes: float64(11), int64(3)
memory usage: 59.3 KB
"""
newdf = df.drop(['medv'], axis=1)       # 독립변수
y = df['medv']                          # 종속변수


# 독립변수는 모두 수치형 자료이므로 상관계수도 같이 테스트.
corr_ = df.corr(method='pearson')
corr_idx = corr_.index.tolist()

for idx in corr_idx:
    print(f"{idx}와 'medv'의 상관계수 = {corr_[idx]['medv']}")

abs_corr = np.abs(corr_)['medv']
coef = np.argsort(-abs_corr.values)
feature_name = abs_corr.index[coef]
print(coef)
print(feature_name[1:])  # 첫 컬럼은 'medv' 자신의 상관계수 1.0이므로 제외

X = newdf.copy()
scaler = StandardScaler()
scaled_x = scaler.fit_transform(X)

trainx, testx, trainy, testy = train_test_split(scaled_x, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(trainx, trainy)

# 예측값
pred = model.predict(testx)


# 1. r2-Score
# sklearn의 r2는 항상  "1 - sse/sst"로 계산한다.
r2 = r2_score(testy, pred)

sse = np.sum( (testy - pred)**2 )
ssr = np.sum( (pred - np.mean(testy))**2 )
sst = np.sum( (testy - np.mean(testy))**2 )
print(sse)
print(ssr)
print(sst)
print(1-sse/sst)
print(r2)


# 2. MSE (Mean Squared Error)
mse = mean_squared_error(pred, testy)
mse2 =  np.mean((testy - pred) ** 2)
print(mse)
print(mse2)


# 3. RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)
rmse2 = mse ** 0.5
rmse3 = mean_squared_error(testy, pred, squared=False )
print(rmse)
print(rmse2)
print(rmse3)


# 4. MAE (Mean Absolute Error)
mae = mean_absolute_error(testy, pred)
mae2 = np.mean(np.abs(testy-pred))
print(mae)
print(mae2)


# 5. MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(testy, pred)
mape2 = np.mean(np.abs((testy-pred)/testy))
print(mape)
print(mape2)



# # MSE 성능 지표를 계산하는 다양한 방법
# rmse = np.sqrt(mse)
# rmse2 = np.sqrt(((testy-pred)**2).mean())
# rmse3 = mean_squared_error( testy, pred, squared=False )
# print(rmse, rmse2, rmse3)
#
#
