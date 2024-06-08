
"""
 회귀분석 (Regression Analysis)
    - 변수 사이의 인과관계를 규명하는 통계분석.
    - 독립변수 : 다른 변수에 영향을 주는 원인.
        ▶ 설명(Explanatory) or 예측(Predictor) 변수.
    - 종속변수 : 변수에 의해 영향을 받는 결과.
        ▶ 반응(Response) or 결과(Outcome) 변수.

    ★ 회귀분석 모델은 독립/종속 모두 "등간척도" or "비율척도"인 [연속형] 변수.
"""

"""
 단순회귀분석
    - 독립/종속 변수가 1개.
    - 선형 방정식(Linear Equation)의 회귀식으로 분석.
"""

# ols = Ordinary Least Squares  (최소 제곱, 최소 자승)
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('./workpython/women.csv'
                      , index_col=0
                      , encoding='euc-kr'
                      , header=0)
print(dataset.columns)

x = dataset['height']   # 키 (독립변수)
y = dataset['weight']   # 몸무게 (종속변수)
"""
    statsmodels : 통계모델 구축.
    formula : R 스타일의 공식을 사용한 모델 정의 관련.
        - R 스타일 공식 예시 = 'y ~ x1 + x2'는 종속/독립 변수의 관계를 지정.
"""
fit = ols('y ~ x', data=dataset).fit() #선형회귀분석 모델.
print(fit.summary()) #분석 결과 요약


# 1. Y절편 추정값
print(f"y절편 추정값: {fit.params.Intercept}")

# 2. 기울기 추정값
print(f"기울기 추정값: {fit.params.x}")

# 3. 독립변수에 따른 종속변수 추정값
print("### 독립변수(키)에 따른 종속변수(몸뭄게)의 추정값 ###")
print(fit.fittedvalues)

# 4. 잔차값 (Residuals)
print(f"###  잔차값  ###")
print(f"{fit.resid}")
print(f"잔차값 평균 : {fit.resid.mean()}")


# 예측1
print("### 새로운 변수 값(키 80 inches)에 대한 몸무게(pounds) 예측값 ###")
print(fit.predict(exog=dict(x=65)))  # exogeneous : 외생변수

# 예측2
# Row에서 특정 독립변수(키)에 대한 값과 종속변수의 실제값, 그리고 예측값.
print(f"5번 Row의 키 값={dataset.iloc[4, 0]},"             # iloc는 0-based-index
      f" 종속변수 몸무게 값 = {dataset.loc[5,'weight']}")   #   loc는 1-based-index

# 키에 대한 몸무게 예측값
print(f"키:[{dataset.iloc[4, 0]}]에 대한 몸무게 예측값: "
      f"{fit.predict(exog=dict(x=dataset.iloc[4,0])).values}")

# 예측값 직접 계산 ( Intercept: y절편값,  params.x: 기울기 )
rst = fit.params.Intercept + dataset.iloc[4,0] * fit.params.x

# 실제값과 예측값의 상대오차
relative_error = (dataset.iloc[4,1] - fit.predict(exog=dict(x=dataset.iloc[4,0])).values)\
    /dataset.iloc[4,1] *100

print(f"실제값과 예측값의 차이인 상대오차의 절댓값: {abs(relative_error)}")


plt.figure(figsize=(10,5))  # 그래프 도면 크기 설정.
plt.scatter(x, y)  # 산점도 그래프
plt.plot(x, fit.fittedvalues, color='red')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.legend( ('Actual Value', 'Predictive Value'), loc='center right')
plt.show()













