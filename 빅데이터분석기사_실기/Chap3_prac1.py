"""
보스톤 하우징 데이터
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

df= pd.read_csv('./workpython/Boston.csv', encoding='euc-kr',
                index_col=0)

print(df.head())
df.info()
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


