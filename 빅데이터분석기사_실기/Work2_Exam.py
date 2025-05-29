
"""
    작업 2유형

    ※ 데이터 분석 순서
        1. 라이브러리 및 데이터 확인
        2. 데이터 탐색(EDA)
        3. 데이터 전처리 및 분리
        4. 모델링 및 성능 평가
        5. 예측값 제출
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# diabetes 데이터셋 Load
ds = load_diabetes()
x  = pd.DataFrame(ds.data, columns=ds.feature_names)
y  = pd.DataFrame(ds.target)


# 실기 시험 데이터셋 세팅
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2,
                                                random_state=2023)

Xtest = pd.DataFrame(Xtest.reset_index())
Xtrain = pd.DataFrame(Xtrain.reset_index())
Ytrain = pd.DataFrame(Ytrain.reset_index())

Xtest.rename(columns={'index':'cust_id'}, inplace=True)
Xtrain.rename(columns={'index':'cust_id'}, inplace=True)
Ytrain.columns = ['cust_id', 'target']

"""
    실제 시험에서는 train, test 데이터 변수를 x, y의 소문자 형식으로 지정.
    X, Y 등의 대문자는 관례상 2차원 이상의 데이터를 다루는 변수로 사용.
    
    데이터 가져오기로 pd.read_csv() 방식 참조. 
"""


"""
    Sample 1. 
        - 당뇨병 환자의 질병 진행정도를 예측.
        - 데이터의 결측치, 이상치, 변수들에 대해 전처리.
        - 회귀모델을 사용해 Rsq, MSE 값을 추출.
        - 제출은 cust_id, target 변수를 가진 dataframe 형태.
"""

# 1. 라이브러리 및 데이터 확인
print(ds.DESCR)
"""
**Data Set Characteristics:**
:Number of Instances: 442
:Number of Attributes: First 10 columns are numeric predictive values
:Target: Column 11 is a quantitative measure of disease progression one year after baseline
:Attribute Information:
    - age     age in years
    - sex
    - bmi     body mass index
    - bp      average blood pressure
    - s1      tc, total serum cholesterol
    - s2      ldl, low-density lipoproteins
    - s3      hdl, high-density lipoproteins
    - s4      tch, total cholesterol / HDL
    - s5      ltg, possibly log of serum triglycerides level
    - s6      glu, blood sugar level
"""
#-----------------------------------------------------------------------------------------------#
# 2. 데이터 탐색 (EDA)
# 데이터 행/열 확인.
print(Xtrain.shape)
print(Xtest.shape)
print(Ytrain.shape)

# 초기 데이터 확인
print(Xtrain.head())
print(Xtest.head())
print(Ytrain.head())

# 변수명과 데이터 타입이 매칭이 되는지, 결측치가 존재하는지 확인.
print(Xtrain.info())
print(Xtest.info())
print(Ytrain.info())

# 기초통계랑을 확인  (Xtrain과 Xtest 간)
print(Xtrain.describe())  # 또는 Xtrain.describe().T
print(Xtest.describe())
print(Ytrain.describe())


#------------------------------------------------------------------------------------#
# 3. 데이터 전처리 & 분리
# 결측치, 이상치, 변수 처리
print(Xtrain.isnull().sum())
print(Xtest.isnull().sum())
print(Ytrain.isnull().sum())

# 결측치 제거
# df = df.dropna()

# 결측치 대체
# 연속형 변수는 중앙, 평균값 등으로,
# 범주형 변수는 최빈값 등으로
# df['variable'] = df['variable'].fillna(대체값)

# 이상치 대체
# df['variable'] = np.where( df['variable'] >= 조건, 대체값, df['variable'] )


# @@@@ 변수 처리 @@@@
# 불필요한 변수 제거
#  df = df.drop( columns= ['변수1', '변수2'] )
# df = df.drop( ['변수1', '변수2', '변수3'], axis=1 )

# 필요 시 변수 추가 (파생 변수 등)
# df['파생변수명'] = df['A'] * df['B']   <-- 파생변수 식은 의도에 맞게 작성

# One-Hot Incoding
# Xtrain = pd.get_dummies(Xtrain)
# Xtest = pd.get_dummies(Xtest)

cust_id = Xtest['cust_id'].copy()

# 각 데이터에서 cust_id 제거
Xtrain = Xtrain.drop( columns= ['cust_id'] )
Xtest =  Xtest.drop(['cust_id'], axis=1)



# 데이터 분리  (훈련 세트와 검증 세트로 분할  80% 훈련, 20% 검증)
x_train, x_val, y_train, y_val = train_test_split(Xtrain, Ytrain['target'],
                                                  test_size=0.2,
                                                  random_state=23)

print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

#------------------------------------------------------------------------------------#
# 4. 모델링 및 성능 평가
# 랜덤포레스트 모델 ( 분류: RandomForestClassifier.  회귀: RandomForestRegressor )
model = RandomForestRegressor(random_state=2023)
model.fit(x_train, y_train)

# 모델을 사용해 테스트 데이터 예측
y_pred = model.predict(x_val)

# 모델 성능 평가 (평균 제곱 오차(MSE) 및 R-squared)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
rmse = mse**0.5  #RMSE
print(mse)
print(r2)

#------------------------------------------------------------------------------------#
#   5. 예측값 제출
# ★ x_test를 model에 넣어서 나온 예측 결과를 제출해야 한다.


# model을 사용해 테스트 데이터 예측
y_rst = model.predict(Xtest)
result = pd.DataFrame({'cust_id': cust_id, 'target': y_rst})
print(result[:5])


"""
    MSE : Mean Squared Error (평균 제곱 오차)
        - 예측값과 실제값의 차이를 제곱하고, 평균한 값.
        - 오차의 크기는 모델의 예측이 실제와 얼마나 벗어났는지 측정하는 지표.
        - 값이 클수록 많이 벗어났음을 의미.
        
    R^2 Score
        - 회귀모델이 실제 데이터의 분산을 얼마나 잘 설명하는지 나타냄.
        - 모델이 전체 변동성 중 얼마나 많은 부분을 설명할 수 있는지 판단.
        - 0 ~ 1 사이의 값을 가짐.
        - 0에 가까울수록 모델이 데이터를 정확히 설명하지 못하는 의미.
        - 1에 가까울수록 모델이 데이터를 잘 설명한다는 의미.
        - 음수라면 예측을 신뢰할 수 없음을 의미.
"""











