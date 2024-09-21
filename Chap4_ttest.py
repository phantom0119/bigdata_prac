# 평균차이 분석 실습
# - 수치형 자료 집단의 차이를 검정하는 방법.

import pandas as pd
import numpy as np
from scipy import stats
import sys

df = pd.read_csv("workpython/data.csv", encoding='euc-kr', index_col=0)
sys.stdout.write(f"DataFrame 구성 컬럼 목록\n{df.columns}")
sys.stdout.write("\n-----------------------------------------------------------------\n")

# 분석 주제
# - 성별(남,여)에 따른 쇼핑 금액의 차이
# - 평균차이 분석: 두 집단 사이의 평균 차이가 유의한지 분석.
#   t-test

#1 성별(남,여) 기준으로 행(Row)을 나누어서 2개의 DataFrame에 담기.
sys.stdout.write(f"성별 unique 항목: {df['성별'].unique()}")
sys.stdout.write("\n-----------------------------------------------------------------\n")

dmale = df[df['성별'] == '남자']
dfemale = df[df['성별'] == '여자']

#성별이 기록되지 않은 (None) 내역 확인.
sys.stdout.write(f"성별이 None인 Row\n{df[df['성별'].isnull()]}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


#쇼핑액이 기록되지 않은 내역 확인.
sys.stdout.write(f"남자 중 쇼핑액이 누락된 Row\n{dmale[dmale['쇼핑액'].isnull()]}\n")
sys.stdout.write(f"여자 중 쇼핑액이 누락된 Row\n{dfemale[dfemale['쇼핑액'].isnull()]}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


sys.stdout.write(f"남자의 쇼핑액 평균 = {np.mean(dmale.쇼핑액)}\n")
sys.stdout.write(f"여자의 쇼핑액 평균 = {np.mean(dfemale.쇼핑액)}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


#2 남자와 여자의 쇼핑액 분석을 위한 t-test 검정 수행.
"""
    stats.ttest_ind = 독립 표본 t-검정
        -> 독립된 두 집단의 평균을 비교. 두 집단의 차이가 통계적으로 유의한지 분석하는 방법.
    equal_var 옵션 = True/False : 두 집단의 분산이 같은지 다른지 명시 (False=다름)
    
    return 값
      0 idx = t-statistic : 두 집단 간의 평균 차이의 크기를 계산.
      1 idx = p-value : 두 집단 간 평균의 차이가 통계적으로 유의한지 판단하는 기준.
         -> 보편적으로 0.05 (95%) 보다 작으면 귀무 가설(두 집단 간 차이가 없다)을 기각(인정하지 않음/부정)하여 차이가 유의미하단 결론.
"""
t, pvalue = stats.ttest_ind(dmale['쇼핑액'], dfemale['쇼핑액'], equal_var=False)
sys.stdout.write(f" @@ 양측 검정 @@\n")
sys.stdout.write(f"t-statistic (t 검정통계량) : {t}\n")
sys.stdout.write(f"p-value (유의수준 5% /확률) : {pvalue}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


# 유의수준 5% 시 p-value가 0.36 (약 36%) 수준이므로
# 0.05 미만의 확률이 아니므로 귀무가설(두 집단 간 차이가 없다)을 채택한다.
# 만약 0.02 (약 2%) 수준 등의 결과가 나왔다면 귀무가설을 기각하게 되어 두 집단 간 차이가 유의미하단 결론이 나온다.



#3 95%의 신뢰구간 구하기
"""
    자유도 ( Degree of Freedom )
    - 표본 데이터에서 독립적으로 변할 수 있는 변수의 수
    - 두 집단간 평균의 차이를 계산할 때, 자유도가 클수록 t-검정 신뢰도가 높아진다.
"""
COEF_LEVEL = 0.95  #신뢰구간 (95%)
DOF = len(dmale['쇼핑액']) + len(dfemale['쇼핑액']) - 2  #자유도 (Degree of Free)
ALPHA = 1 - COEF_LEVEL  #유의수준 (5%)

# 남성과 여성의 쇼핑액 평균
m = np.mean(dmale['쇼핑액'])
f = np.mean(dfemale['쇼핑액'])
meandiff = m-f

# 표준오차 (표본의 표준편차를 표본의 크기로 나눔) 계산
stderror = np.sqrt(np.var(dmale['쇼핑액']) / len(dmale['쇼핑액'])
                +  np.var(dfemale['쇼핑액']) / len(dfemale['쇼핑액']))

# 허용 오차 : t-분포의 임계값을 계산한다.
marginerror = stats.t.ppf( 1-ALPHA/2, DOF) * stderror

# 신뢰구간
conf_interval = (meandiff-marginerror, meandiff+marginerror)
sys.stdout.write(f" @@ 양측 검정 신뢰구간 @@\n")
print(conf_interval)
sys.stdout.write("\n-----------------------------------------------------------------\n")


############################################################################################
# 단측 검정으로 접근하기
"""
    t-test 검정은 '양측'과 '단측' 방식의 접근이 가능하다.
    
    -양측
     > 대립: 두 집단의 평균이 서로 다르다 (차이가 있는지의 여부만 확인)
     > 귀무: 두 집단의 평균이 같다.
     
    -단측 (우측)
     > 대립: 첫 집단의 평균이 두 번째 집단의 평균보다 크다.
     > 귀무: 첫 집단의 평균이 두 번째 집단의 평균보다 작거나 같다.
     
    -단측 (좌측)
     > 대립: 첫 집단의 평균이 두 번째 집단의 평균보다 작다.
     > 귀무: 첫 집단의 평균이 두 번째 집단의 평균보다 크거나 같다.

    -- 단측의 경우 대립 가설의 방향성이 나타나는 것을 볼 수 있다.
"""

# 주제1. 우측 검정
# - 남성의 쇼핑액 평균은 여성의 쇼핑액 평균보다 크다.


# 단측검정 옵션 = alternative
# - greater: x, y 집단 입력 순서 기준으로 x 집단이 y 집단보다 평균적으로 큰지 확인.
# - less: x, y 집단 입력 순서 기준으로 x 집단이 y 집단보다 평균적으로 작은지 확인.
# - two-sided: 양측검정 (기본값)
t, pvalue = stats.ttest_ind(dmale['쇼핑액'], dfemale['쇼핑액'],
                            equal_var=False,
                            alternative='greater')

sys.stdout.write(f" @@ 단측 검정 (우측) @@\n"
                 f"대립 가설: 남성의 평균 쇼핑액이 여성의 평균 쇼핑액보다 크다.\n"
                 f"귀무 가설: 남성의 평균 쇼핑액이 여성의 평균 쇼핑액보다 작거나 같다.\n")

sys.stdout.write(f't-statistic : {t}\n')
sys.stdout.write(f'p-value : {pvalue}')
sys.stdout.write("\n-----------------------------------------------------------------\n")













