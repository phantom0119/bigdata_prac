
import pandas as pd
import numpy as np

df= pd.read_csv('./workpython/Boston.csv', encoding='euc-kr',
                index_col=0)

print(df.head())
# 상관계수 확인
"""
     DataFrame.corr() : 데이터프레임의 각 열 쌍 사이의 "피어슨 상관 계수" 계산.
        - 두 변수 간의 선형 관계의 강도와 방향을 나타냄.
        -     -1 ~ 1 사이의 값을 가짐.
        -   1에 가까울수록 강한 양의 상관관계
        -  -1에 가까울수록 강한 음의 상관 관계
        - 0이라면 전혀 선형 관계가 없음.
        
    0.8 이상 또는 -0.8 이하인 경우 "다중공선성"을 의심.
"""
print(df.corr())
corr_mat = df.corr()

# 상삼각 행렬로 만들기 위한 전처리 작업.
# 하단부는 NaN으로 변환
upper_corr_mat = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
print(upper_corr_mat)

# 상감각 행렬에서 가장 큰 상관계수 찾기
print(upper_corr_mat.stack().max())


