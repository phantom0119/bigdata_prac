# 다원배치 분산분석 실습 (반복 측정 분산 분석)
"""
- 표본의 수가 동일한 집단에 대한 비교.
- 주로 동일 집단의 세 가지 이상의 조건에 대한 측정 결과 분석 시 사용.
- ex. 100명의 고객에 대한 (1월, 2월, 3월) 쇼핑액이 월별로 차이가 있는지에 대한 분석.
"""
# statsmodels.formula.api
# statsmodels.api


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
import sys
warnings.filterwarnings('ignore')


df = pd.read_csv('workpython/data.csv', encoding='euc-kr', index_col=0)
sys.stdout.write(f"@@ 데이터프레임 구성 컬럼 리스트\n{df.columns}")
sys.stdout.write("\n-----------------------------------------------------------------\n")

# 분석 주제: 고객별로 (쇼핑1월, 쇼핑2월, 쇼핑3월) 월별 쇼핑액의 차이가 있는지 검정.

#1 분석에 필요한 데이터 컬럼만 가져와 새로운 데이터프레임으로 구성.
# - 새로운 컬럼을 추가하고자 한다면 데이터프레임 구조에서 작업해야 한다.
# - Series 객체에서는 열(Column) 추가가 되지 않기 때문이다.
newdf = df[['쇼핑1월', '쇼핑2월', '쇼핑3월']]

# 각 데이터(Row)별 월 구분을 위한 새로운 Column 추가 (데이터프레임 객체에서 작업).
newdf['month1'] = 1
newdf['month2'] = 2
newdf['month3'] = 3

#print(newdf)
#print(newdf.shape)
"""
        쇼핑1월  쇼핑2월  쇼핑3월  month1  month2  month3
고객번호                                            
190105  76.8  64.8  54.0       1       2       3
190106  44.4  32.4  39.6       1       2       3
190107  66.0  66.0  51.6       1       2       3
190108  62.4  52.8  52.8       1       2       3
190109  63.6  54.0  51.6       1       2       3
...      ...   ...   ...     ...     ...     ...
190190  88.8  90.0  52.8       1       2       3
190191  51.6  88.8  27.6       1       2       3
190192  88.8  88.8  38.4       1       2       3
190193  90.0  90.0  25.2       1       2       3
190194  75.6  74.4  67.2       1       2       3

[90 rows x 6 columns]
(90, 6)   <- 행,렬
"""

# 월별 데이터를 Row 단위로 추출하기 위한 피벗 변환 목적의 데이터프레임 설계
dm1 = newdf[['쇼핑1월', 'month1']] # 1월 쇼핑액 Row 단위 데이터 추출
dm2 = newdf[['쇼핑2월', 'month2']] # 2월 쇼핑액 Row 단위 데이터 추출
dm3 = newdf[['쇼핑3월', 'month3']] # 3월...

# inplace 옵션 = 데이터프레임에서 변경된 내용을 즉시 덮어쓰기
dm1.rename(columns={'쇼핑1월': 'value', 'month1': 'month'}, inplace=True)    # 데이터 통합을 위한 컬럼명 통일
dm2.rename(columns={'쇼핑2월': 'value', 'month2': 'month'}, inplace=True)
dm3.rename(columns={'쇼핑3월': 'value', 'month3': 'month'}, inplace=True)

# 데이터 통합
# ignore_index = 새로운 인덱스로 통합하기 (0부터 재할당)
dmset = pd.concat([dm1, dm2, dm3], ignore_index=True)
print(dmset)
sys.stdout.write("\n-----------------------------------------------------------------\n")



#2 분석 작업 수행
# - value 위치에는 종속 변수 Column을 작성.
# - month 위치에는 독립 변수 Column을 작성.
# 여기서는 C()를 통해 '범주형 변수'로 취급되며 월 범주형 값을 담은 'month' Column을 독립 변수로 선택.
# 월 구분에 대한 쇼핑액(만원단위) 값을 담은 'value' Column을 종속 변수로 선택.

# OLS = Ordinary Least Squares : RSS (Residual Sum of Squares 잔차제곱합) 최소화.
model = ols('value ~ C(month)', data=dmset).fit()
# 회귀 모델을 적합(fit) --> 회귀 계수와 통계량을 계산해 model 객체로 담는다.


"""
회귀계수: model.params
pvalue: model.pvalues
결정계수 (R-squared): model.rsquared
잔차(Residuals): model.resid
예측값: model.fittedvalues
"""
print(model.summary())
sys.stdout.write("\n-----------------------------------------------------------------\n")

# ANOVA 분석 결과 테이블 생성
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
sys.stdout.write("\n-----------------------------------------------------------------\n")


sys.stdout.write(f"ANOVA 분석 F-통계량 = {anova_table.loc['C(month)']['F']}\n")
sys.stdout.write(f"ANOVA 분석 pvalue = {anova_table.loc['C(month)']['PR(>F)']}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


#3 가설 검정
ALPHA = 0.5 #유의수준 5%
if anova_table.loc['C(month)']['PR(>F)'] < ALPHA:
    print("결론: 귀무가설 기각\n"
          "--> 고객별로 월별 쇼핑액 간의 유의미한 차이가 있다.")
else:
    print("결론: 귀무가설 채택\n"
          "-->고객별로 월별 쇼핑액 간의 유의미한 차이가 없다.")






