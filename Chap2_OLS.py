
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
import pandas as pd
import numpy as np

dataset = pd.read_csv('./workpython/women.csv'
                      , index_col=0
                      , encoding='euc-kr'
                      , header=0)
#print(dataset.columns)
"""
Index(['height', 'weight'], dtype='object')
"""

x = dataset['height']   # 키 (독립변수)
y = dataset['weight']   # 몸무게 (종속변수)
"""
    statsmodels : 통계모델 구축.
    formula : R 스타일의 공식을 사용한 모델 정의 관련.
        - R 스타일 공식 예시 = 'y ~ x1 + x2'는 종속/독립 변수의 관계를 지정.
"""

# ols 모델 적용
model = ols('y ~ x', data=dataset).fit()
#print(model.summary())
"""
 OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.991
Model:                            OLS   Adj. R-squared:                  0.990
Method:                 Least Squares   F-statistic:                     1433.
Date:                Wed, 09 Oct 2024   Prob (F-statistic):           1.09e-14
Time:                        00:48:38   Log-Likelihood:                -26.541
No. Observations:                  15   AIC:                             57.08
Df Residuals:                      13   BIC:                             58.50
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -87.5167      5.937    -14.741      0.000    -100.343     -74.691
x              3.4500      0.091     37.855      0.000       3.253       3.647
==============================================================================
Omnibus:                        2.396   Durbin-Watson:                   0.315
Prob(Omnibus):                  0.302   Jarque-Bera (JB):                1.660
Skew:                           0.789   Prob(JB):                        0.436
Kurtosis:                       2.596   Cond. No.                         982.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# 결정계수 (R-squared) 확인
# print(model.rsquared)
# 0.9910098326857506


# DataFrame의 10번 Row에 대한 값을 확인 후, model의 x값 예측 결과랑 비교하기
#print(dataset.iloc[10])
"""
height     68
weight    146
Name: 11, dtype: int64
"""

# model 예측 진행
# 외생변수 명인 'x'는 모델에서 사용된 독립변수명과 일치해야 한다.
# --> ols 모델에서 'y ~ x'였으므로 외생변수명은 x가 되어야 한다.
testm = model.predict(exog=dict(x=68))    # exogeneous:외생변수
#print(testm)
"""
0    147.083333     -->  0: 인덱스 번호
dtype: float64
"""

# DataFrame 자료의 모든 독립변수에 대한 추측(예측)-fitted 값-values을 출력.
fittedvalue = model.fittedvalues
#print(fittedvalue)
"""
1     112.583333
2     116.033333
3     119.483333
4     122.933333
5     126.383333
6     129.833333
7     133.283333
8     136.733333
9     140.183333
10    143.633333
11    147.083333
12    150.533333
13    153.983333
14    157.433333
15    160.883333
dtype: float64
"""

# DataFrame 자료의 모든 독립변수에 대한 잔차(Residual) 출력.
# Residual: 관측(실제)값 - 예측값   --> 값이 작을수록 설명력이 좋은 것.
resid = model.resid
#print(resid)
"""
1     2.416667
2     0.966667
3     0.516667
4     0.066667
5    -0.383333
6    -0.833333
7    -1.283333
8    -1.733333
9    -1.183333
10   -1.633333
11   -1.083333
12   -0.533333
13    0.016667
14    1.566667
15    3.116667
dtype: float64
"""

#  DataFrame 자료의 모든 독립변수에 대한 상대오차(Relative Error) 출력.
#  상대오차 = (잔차 / 실제값)의 절대값
redf = pd.DataFrame(resid, columns=['residual'])

# 모든 독립변수에 대한 상대오차 List 생성
relerr = list()
for idx, resiv in enumerate(resid):
    #print(idx, round(resiv, 6), x.iloc[idx])
    value = abs(resiv)/abs(y.iloc[idx])
    #print(value)
    relerr.append(value)

relerr = np.array(relerr)
redf['RelativeError'] = relerr
redf['x'] = x
redf['y'] = y


#print(redf)
"""
    residual  RelativeError   x    y
1   2.416667       0.021014  58  115
2   0.966667       0.008262  59  117
3   0.516667       0.004306  60  120
4   0.066667       0.000542  61  123
5  -0.383333       0.003042  62  126
6  -0.833333       0.006460  63  129
7  -1.283333       0.009722  64  132
8  -1.733333       0.012840  65  135
9  -1.183333       0.008513  66  139
10 -1.633333       0.011502  67  142
11 -1.083333       0.007420  68  146 
12 -0.533333       0.003556  69  150
13  0.016667       0.000108  70  154
14  1.566667       0.009853  71  159
15  3.116667       0.019004  72  164
"""
