# 회귀분석 모델로 예측 모델 만들기 실습


import pandas as pd
import numpy as np
import sys
import math
from statsmodels.formula.api import ols


"""
header 옵션: 첫 행에 컬럼 값이 있음
index_col 옵션: 해당 인덱스의 Column을 인덱스 열로 적용.
"""
df = pd.read_csv('workpython/mtcars.csv', encoding='utf-8',
                 header=0, index_col=0)

sys.stdout.write(f"데이터프레임 컬럼 구조\n"
                 f"{df.columns}")
print("\n--------------------------------------------------------------------")


sys.stdout.write(f"데이터프레임 컬럼별 데이터 타입\n"
                 f"{df.dtypes}")
print("\n--------------------------------------------------------------------")
sys.stdout.write(f"{df[['disp', 'hp', 'wt', 'qsec']].head(20)}")
print("\n--------------------------------------------------------------------")



# 특정 4개의 변수를 독립변수로 구성,   'mpg' Column은 종속변수로 구성
# disp: 배기량(cc)             float64 (부동소수점)
# hp : 마력                   int64 (정수)
# wt: 무게(1000ibs)           float64 (부동소수점)
# qsec : 1/4마일 도달 시간      float64 (부동소수점)

x = df[['disp', 'hp', 'wt', 'qsec']]
y = df['mpg']

model = ols('y ~ x', data=df).fit()   # 다중 선형회귀 분석 모델


# 데이터프레임에 새 컬럼 값으로 예측 결과 값을 추가
df['pred'] = model.fittedvalues

sys.stdout.write(f"{df[['disp', 'hp', 'wt', 'qsec', 'mpg', 'pred']].head(20)}")
print("\n--------------------------------------------------------------------")



# 성능평가 지표

me = (df['mpg'] - df['pred']).mean()

print(f"평균 예측 오차\n"
      f"(실측-예측)의 평균 계산\n"
      f"{me}\n")

mse = ((df['mpg']-df['pred']) * (df['mpg']-df['pred'])).mean()

print(f"평균 제곱 오차\n"
      f"(실측-예측)^2의 평균 계산\n"
      f"{mse}\n")


rmse = math.sqrt(mse)

print(f"평균 제곱근 오차 (Root Mean of Squared Errors)\n"
      f"평균 제곱 오차의 제곱근(sqrt)\n"
      f"{rmse}\n")

mae = abs(df['mpg']-df['pred']).mean()

print(f"평균 절대 오차\n"
      f"(실측-예측)의 절댓값의 평균\n"
      f"{mae}\n")

mpe = ((df['mpg']-df['pred'])/df['mpg']).mean()

print(f"MPE (Mean Percentage Error)\n"
      f"평균 백분오차 : (실측-예측)값을 실측 값으로 나눈 값들의 평균\n"
      f"{mpe}\n")


mape = abs(((df['mpg']-df['pred'])/df['mpg'])).mean()

print(f"MAPE (Mean Absolute Percentage Error\n"
      f"평균 절대 백분오차\n"
      f"{mape}\n")

print("\n--------------------------------------------------------------------")

# 모델의 summary 추출
print(model.summary())

