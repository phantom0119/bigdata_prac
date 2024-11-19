# 비율차이 분석 실습
# - 이항형 자료 (예, 아니요), (합, 불) 간의 차이 검정 시 사용하는 분석 방법.

import pandas as pd
import numpy as np
from scipy import stats
import sys

df = pd.read_csv("workpython/data.csv", encoding='euc-kr', index_col=0)
sys.stdout.write(f"데이터프레임 구성 컬럼\n{df.columns}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


# 분석 주제: 성별(남/여)에 따른 쿠폰 선호도의 비율 차이가 존재하는지 확인.

#1 남성, 여성 집단 분리하기
sys.stdout.write(f"'성별' 컬럼의 데이터 도메인 확인\n{df['성별'].unique()}")
sys.stdout.write("\n-----------------------------------------------------------------\n")

dmale = df[df['성별'] == '남자']
dfemale = df[df['성별'] == '여자']

# 남성, 여성의 쿠폰 선호도 도메인 확인.
sys.stdout.write(f"남성의 쿠폰 선호도 도메인 = {dmale['쿠폰선호도'].unique()} \n")
sys.stdout.write(f"여성의 쿠폰 선호도 도메인 = {dfemale['쿠폰선호도'].unique()}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


#2 비율 검정에 필요한 쿠폰 선호도 이항 데이터 수집 (예, 아니오 분리) -> observed
# 남성의 쿠폰 선호도 '예' 개체 수 = my.  여성은 fy
my = len(dmale[dmale['쿠폰선호도'] == '예'])
fy = len(dfemale[dfemale['쿠폰선호도'] == '예'])

observed = [[my, fy], [len(dmale)-my, len(dfemale)-fy]]

sys.stdout.write(f"[남성 전체 수, 여성 전체 수] = [{len(dmale)}, {len(dfemale)}]\n")
sys.stdout.write(f"[남성 쿠폰 선호도 Y, 여성 쿠폰 선호도 Y] = {observed[0]}\n")
sys.stdout.write(f"[남성 쿠폰 선호도 N, 여성 쿠폰 선호도 N] = {observed[1]}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


#3 카이제곱 검정을 통한 비율 분석 진행.
"""
    stats.chi2_contingency()
     - 카이제곱 독립성 검정 수행 함수
    
    return
     - chi: 카이제곱 통계량 (Chi-squared statistic) = 관찰된 데이터와 기대값이 얼마나 차이나는지에 대한 통계.
     - pvalue: 가설의 유의성 판별 목적의 값.
     - dof: 자유도. 행렬의 크기와 관련.
     - expect: 기대값 행렬 (Expected Frequencies) = 두 집단이 독립적이라고 가정했을 때, 각각의 셀에서 기대되는 빈도.
"""
chi, pvalue, dof, expect = stats.chi2_contingency(observed)
sys.stdout.write(f"카이제곱 검정 통계량 = {chi}\n")
sys.stdout.write(f"pvalue(유의확률) = {pvalue}\n")
sys.stdout.write(f"기대 빈도수 = {expect}\n")
sys.stdout.write(f"자유도 = {dof}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


#4 가설 검정
ALPHA = 0.05 #유의수준 5%
if pvalue < ALPHA:
    print("결론: 귀무가설 기각\n--> 비율의 차이가 유의미하게 존재: 성별에 대한 쿠폰 선호도의 비율 차이가 있다.")
else:
    print("결론: 귀무가설 채택\n--> 비율의 차이가 없다: 성별에 대한 쿠폰 선호도의 비율 차이가 없다.")










