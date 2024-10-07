# 상관관계 실습

import pandas as pd
from scipy import stats


df = pd.read_csv('workpython/women.csv', header=0, index_col=0)
#df.info()
"""
<class 'pandas.core.frame.DataFrame'>
Index: 15 entries, 1 to 15
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   height  15 non-null     int64
 1   weight  15 non-null     int64
dtypes: int64(2)
memory usage: 360.0 bytes
"""

#print(df.cov())
"""
        height      weight
height    20.0   69.000000
weight    69.0  240.209524

1. height-height = 20 (height의 분산은 20)
  -- 분산: 평균으로부터 얼마나 떨어져 있는지를 나타내는 지표
          값이 클 수록 변동성이 크다는 것을 의미
          
2. weight-weight = 240 (weight의 분산은 240 수준)
  -- 값이 크므로 변동성이 상당히 높음을 의미

3. height-weight = 69  (height와 weight의 공분산)
  -- 양의 값을 가지고 있으므로 하나의 변수가 증가할 때 다른 변수도 증가하는 경향을 보인다.
  -- 값이 큰 편이므로 강한 상관관계가 있다고 판단할 수 있다.

"""

# 상관계수 종류별 테스트
#print(df.corr(method='pearson'))
"""
          height    weight
height  1.000000  0.995495
weight  0.995495  1.000000
"""


# 스피어만: 단조적이지만 선형적이지 않은 관계 측정
#print(df.corr(method='spearman'))
"""
        height  weight
height     1.0     1.0
weight     1.0     1.0
"""


# 캔달: 순위 기반의 관계 측정
#print(df.corr(method='kendall'))
"""
        height  weight
height     1.0     1.0
weight     1.0     1.0
"""


# scipy 라이브러리의 stats 모듈에서 pearson 상관계수 적용.
pearson = stats.pearsonr(df['height'], df['weight'])
#print(pearson, pearson[1])
"""
PearsonRResult(statistic=0.9954947677842163, pvalue=1.0909729585995878e-14)

pearson[0] = statistic 값 (T-검정통계량)
pearson[1] = pvalue (유의 확률)
"""