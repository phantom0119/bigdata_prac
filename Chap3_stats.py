
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('./workpython/data.csv',
                 encoding='euc-kr',
                 index_col=0)
dfmale = df[df['성별'] == '남자']
dfemale = df[df['성별'] == '여자']

m = np.mean(dfmale['쇼핑액'])      # 남자 쇼핑액
f = np.mean(dfemale['쇼핑액'])     # 여자 쇼핑액

conf_level = 0.95       # 신뢰구간 상수 (95%)
alpha = 1 - conf_level  # 유의수준 (5%)

# 자유도는 각 표본의 크기에서 1을 뺀 값이다.
# 따라서 ( 남성 쇼핑액 수 -1 ) + ( 여성 쇼핑액 수 -1 ) 의 계산에서 도출된 값이다.
dof = len(dfmale['쇼핑액']) + len(dfemale['쇼핑액']) - 2   # 자유도


"""
    신뢰구간 계산 과정
    필요 자원:
            1. 자유도 =  n1 + n2 - 2
            2. 하한값 = (n1평균 - n2평균) - stats.ppf( 1 - alpha/2, dof ) * 표준에러
"""

meandiff = m - f  # 쇼핑액 평균 차이
stderror = np.sqrt( np.var(dfmale['쇼핑액']) / len(dfmale['쇼핑액'])) +\
    np.var(dfemale['쇼핑액'] / len(dfemale['쇼핑액']))

marginerror = stats.t.ppf( 1-alpha/2, dof) + stderror

conf_interval = ( meandiff-marginerror, meandiff+marginerror )
print(f" (남자-여자) 쇼핑액의 차이에 대한 신뢰구간(95%): {conf_interval}")



"""
    귀무가설= 남자 쇼핑액 >= 여자 쇼핑액. 성별에 따른 쇼핑액의 차이가 없다.
    대립가설= 남자 평균 쇼핑액이 여자 평균 쇼핑액보다 작다. ( 남자 쇼핑액 < 여자 쇼핑액 )
"""
t, p = stats.ttest_ind( dfmale['쇼핑액'], dfemale['쇼핑액'],
                        equal_var=False,
                        alternative='less')

"""
    독립표본 t 검정 = 검정통계량(t분포),  p-value = 유의확률
    - equal_var = 등분산 ( True: 두 집단의 분산이 같음. 기본값,  False: 다름 )
    - alternative =  less: 대립가설 (남자<여자, 남자의 평균 쇼핑액이 여자보다 작다. )
"""

print(f"t-검정통계량: {t}")
print(f"유의확률 p-value: {p}")




# ------------------- 카이제곱 검정 -------------------------
nomale = len(dfmale)
nofemale = len(dfemale)

x1 = len(dfmale[dfmale['쿠폰선호도'] == '예'])    # 남자의 쿠폰선호도
x2 = len(dfemale[dfemale['쿠폰선호도'] == '예'])  # 여자의 쿠폰선호도

# 비율 검정 데이터
observe = [[x1, x2], [nomale-x1, nofemale-x2]]

# 카이제곱 검정 모델
chi, pvalue, dof, expect = stats.chi2_contingency(observe)
print(f"카이제곱 검정 통계량 = {chi}\n"
      f"p-value(유의확률) = {pvalue}\n"
      f"기대 빈도수 = {expect}")

alpha = 0.5  # 유의수준 5%
if pvalue < alpha :  # 귀무가설 기각 조건
    print("비율 차이가 유의미하게 존재 (귀무가설 기각)")
else:
    print("비율 차이가 유의미하지 않습 니다. (귀무가설 채택)")



# -------------------   분산 분석  ---------------------
df1 = df[df['주거지역'] == '소도시']
df2 = df[df['주거지역'] == '중도시']
df3 = df[df['주거지역'] == '대도시']

print(f" @@@@@ 주거지역 별 기술통계량 @@@@@\n"
      f"{df1.describe()}\n"
      f"{df2.describe()}\n"
      f"{df3.describe()}")








