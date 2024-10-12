# 다중 선형 회귀 연습
# LinearRegression, RandomForestRegressor, lgb Model로 적용

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgbm

df = pd.read_csv('workpython/seatbelts.csv', index_col=0)
df.info()
"""
Index: 192 entries, 1 to 192
Data columns (total 8 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   DriversKilled  192 non-null    int64       사망자 수 (종속변수)
 1   drivers        192 non-null    int64       사고발생건수
 2   front          192 non-null    int64       앞좌석 승객 수
 3   rear           192 non-null    int64       뒷좌석 승객 수
 4   kms            192 non-null    int64       주행거리
 5   PetrolPrice    192 non-null    float64     휘발유 가격
 6   VanKilled      192 non-null    int64       -미사용
 7   law            192 non-null    int64       -미사용
dtypes: float64(1), int64(7)
memory usage: 13.5 KB
"""

# Column별 데이터 확인
# print(df.iloc[:,1:5].head())
# print(df.iloc[:,5:9].head())
"""
   DriversKilled  drivers  front  rear     kms  PetrolPrice  VanKilled  law
0            107     1687    867   269    9059     0.102972         12    0
1             97     1508    825   265    7685     0.102363          6    0
2            102     1507    806   319    9963     0.102062         12    0
3             87     1385    814   407   10955     0.100873          8    0
4            119     1632    991   454   11823     0.101020         10    0
"""

# 분석 주제 : 주어진 데이터를 바탕으로 사망자 수(DriversKilled) 예측하기.
# Row 기준 데이터 : 1969~1984년 동안의 월별 통계 (16년 x 12개월 = 192 Row)


# 5개의 Column을 독립변수로 사용 : drivers, front, rear, kms, PetrolPrice
# 각 독립변수를 Z-score, Min-Max 표준화 해보고, 각 경우에 대한 LinearRegression 적용하기.


mmScaler = MinMaxScaler()
stScaler = StandardScaler()

newdf = df.copy()

y = newdf['DriversKilled']
newdf = newdf.drop(columns=['DriversKilled', 'VanKilled', 'law'], axis=1)


# Min-Max Scaler
minmaxdf =  mmScaler.fit_transform(newdf)
mm_scaled = pd.DataFrame(minmaxdf, columns=['drivers', 'front', 'rear', 'kms', 'PetrolPrice'])

#print(mm_scaled)
"""
      drivers     front      rear       kms  PetrolPrice
0    0.394490  0.505155  0.106635  0.098558     0.420319
1    0.282405  0.457045  0.097156  0.000000     0.408577
2    0.281778  0.435281  0.225118  0.163403     0.402781
3    0.205385  0.444444  0.433649  0.234560     0.379845
4    0.360050  0.647194  0.545024  0.296822     0.382668
..        ...       ...       ...       ...          ...
187  0.142142  0.249714  0.703791  1.000000     0.648391
188  0.242329  0.248568  0.485782  0.897353     0.634816
189  0.324358  0.246277  0.436019  0.878201     0.680571
190  0.425798  0.326460  0.630332  0.780360     0.672097
191  0.442079  0.337915  0.632701  0.750592     0.672880
"""


# Standard Scaler
stdf = stScaler.fit_transform(newdf)
st_scaled = pd.DataFrame(stdf, columns=['drivers', 'front', 'rear', 'kms', 'PetrolPrice'])

#print(st_scaled)
"""
      drivers     front      rear       kms  PetrolPrice
0    0.057789  0.170527 -1.595072 -2.025194    -0.053705
1   -0.561897 -0.069964 -1.643331 -2.494074    -0.103837
2   -0.565359 -0.178758 -0.991830 -1.716702    -0.128582
3   -0.987715 -0.132950  0.069875 -1.378181    -0.226506
4   -0.132617  0.880549  0.636923 -1.081974    -0.214453
..        ...       ...       ...       ...          ...
187 -1.337371 -1.106368  1.445267  2.263317     0.920035
188 -0.783461 -1.112094  0.335302  1.774985     0.862078
189 -0.329948 -1.123546  0.081940  1.683871     1.057430
190  0.230886 -0.722727  1.071257  1.218404     1.021247
191  0.320896 -0.665467  1.083322  1.076784     1.024591
"""


# Min-Max Scaler 독립변수를 LinearRegression 모델로 학습 후 예측 결과 확인하기
trainx, testx, trainy, testy = train_test_split(mm_scaled, y, test_size=0.2, random_state=2000)

model = LinearRegression()
model.fit(trainx, trainy)

pred = model.predict(testx)
r2 = r2_score(testy, pred)
mse = mean_squared_error(testy, pred)

print(r2, mse)



# Z-Score Scaler 독립변수를 LinearRegression 모델로 학습 후 예측 결과 확인하기
trainx, testx, trainy, testy = train_test_split(st_scaled, y, test_size=0.2, random_state=1000)

model = LinearRegression()
model.fit(trainx, trainy)

pred = model.predict(testx)
r2 = r2_score(testy, pred)
mse = mean_squared_error(testy, pred)

print(r2, mse)

# 결론 : 표준화 모델은 학습 성능에 영향을 주지 않는다 (원하는 모델을 적용하면 된다).


# LGBM 모델로 회귀 모델 만들어보기
# LGBM 학습 용도의 데이터셋 만들기
trainx, testx, trainy, testy = train_test_split(newdf, y, test_size=0.2, random_state=2024)

lgbm_train = lgbm.Dataset(trainx, label=trainy)
lgbm_test = lgbm.Dataset(testx, label=testy)

params = {
    'objective' : 'regression',
    'metric' : 'l2',
    'boosting_type' : 'gbdt',
    'min_gain_to_split': 0.01,
    'learning_rate' : 0.1
}

# lgb = lgbm.train(params, lgbm_train, valid_sets=lgbm_test)
# pred = lgb.predict(testx)
#
# mse = mean_squared_error(pred, testy)
# r2 = r2_score(pred, testy)

#print(r2, mse)

"""
참조 내용 : LGBM 파라미터
1. 기본 파라미터
 > boosting_type: 부스팅 방식 선택
    - gbdt (기본값): 전통적인 그라디언트 부스팅 방식
    - dart: Dropouts meet Multiple Additive Regression Trees
    - goss: Gradient-based One-Side Sampling
    - rf: 랜덤 포레스트 방식


> objective: 손실 함수 (목표 함수)
    - regression: 회귀 분석 (MSE 기반)
    - binary: 이진 분류
    - multiclass: 다중 분류 (클래스 수 설정 필요)
    - lambdarank: 랭킹 모델


> metric: 평가 지표
    - l2 또는 mean_squared_error: 회귀 문제의 MSE
    - binary_logloss: 이진 분류의 로그 손실
    - multi_logloss: 다중 분류의 로그 손실
    - auc: 이진 분류의 AUC



2. 학습 관련 파라미터
learning_rate: 학습률 (기본값: 0.1)
    - 작을수록 학습 속도가 느려지지만 더 세밀한 최적화 가능.

num_leaves: 하나의 트리가 가질 수 있는 리프 노드 수 (기본값: 31)
    - 클수록 모델이 복잡해지고, 과적합 가능성이 높아짐.

max_depth: 트리의 최대 깊이 (기본값: -1, 제한 없음)
    - 트리의 복잡도를 제한하여 과적합을 방지.

min_data_in_leaf: 리프 노드가 최소로 가져야 할 데이터 수 (기본값: 20)
    - 리프 노드에 있는 최소 샘플 수를 지정하여 과적합 방지.

num_iterations 또는 n_estimators: 부스팅 반복 횟수 (기본값: 100)
    - 부스팅 단계 수로, 클수록 성능이 좋아지지만 계산 비용 증가.

early_stopping_rounds: 조기 종료 설정
    - 검증 세트의 성능 향상이 없을 경우, 미리 학습을 중단하는 기능.
    
    
    
3. 정규화 및 과적합 방지 관련 파라미터
bagging_fraction: 트리 학습에 사용할 데이터 샘플 비율 (기본값: 1.0)
    - 1보다 작게 설정하면 데이터의 일부를 무작위로 선택해 사용하여 과적합 방지.


bagging_freq: 배깅을 수행할 빈도 (기본값: 0)
    - bagging_fraction이 설정된 경우, 몇 번째 트리마다 배깅을 적용할지 설정.


feature_fraction: 각 트리 학습에 사용할 피처의 비율 (기본값: 1.0)
    - 1보다 작게 설정하면 일부 피처만 선택해 사용하여 과적합 방지.


lambda_l1 및 lambda_l2: L1 및 L2 정규화 (기본값: 0)
    - 모델의 가중치를 정규화하여 과적합을 방지. 값이 클수록 강한 정규화가 적용됨.
    


4. 트리 학습 관련 파라미터
min_split_gain: 트리가 분할을 수행하기 위한 최소 손실 감소 (기본값: 0.0)
    - 이 값이 클수록 트리가 새로운 분할을 하기 위해 더 큰 손실 감소가 필요하여, 과적합을 방지할 수 있음.


max_bin: 입력 데이터를 이산화할 때 사용할 최대 빈 개수 (기본값: 255)
    - 클수록 데이터의 구체적인 값을 반영하지만, 계산 비용이 증가함.
"""






