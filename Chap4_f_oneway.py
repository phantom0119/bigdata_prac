# 분산분석 실습
# - ANOVA = Analysis of Variance (변량 분석)
# - 3개 이상의 집단에 대해 차이를 비교.
# - 집단 내의 분산, 총평균, 집단의 평균 차이에 의한 집단 간 분산의 비교로
#   F-분포를 사용 (분산의 비교를 통해 얻어진 분포 비율)

"""
    F 분포
    - 각 집단의 모집단 분산이 차이가 있는지, 모집단 평균의 차이가 있는지 검정.
    - (집단 간 변동) / (집단 내 변동)

    일원배치 분산분석
    - 표본의 수가 서로 다른 집단들에 대한 비교.
    - 모집단의 수에 차이가 있어 (소도시, 중도시, 대도시), (1학년, 2학년, 3학년)
      표본의 수를 인위적으로 통제하기 어려운 자료에 대한 차이 분석에 활용.

    반복측정 분산분석
    - 표본의 수가 동일한 집단에 대한 비교.
    - 주로 동일 집단의 세 가지 이상의 조건에 대한 측정 결과 분석 시 사용.
    - ex. 100명의 고객에 대한 (1월, 2월, 3월) 쇼핑액이 월별로 차이가 있는지에 대한 분석.
"""


import pandas as pd
from scipy import stats
import numpy as np
import sys

df = pd.read_csv('workpython/data.csv', encoding='euc-kr', index_col=0)
sys.stdout.write(f"데이터프레임 구성 컬럼\n{df.columns}")
sys.stdout.write("\n-----------------------------------------------------------------\n")

# 분석 주제: 주거지역 별 쇼핑액의 차이가 유의미한지 확인.
#1 주거지역 별로 분류한 데이터셋 확보.
sys.stdout.write(f"주거지역 분류 도메인 = {df['주거지역'].unique()}")
sys.stdout.write("\n-----------------------------------------------------------------\n")

sdf1 = df[df['주거지역'] == '소도시']
sdf2 = df[df['주거지역'] == '중도시']
sdf3 = df[df['주거지역'] == '대도시']

# 각 집단별 쇼핑액 누락 데이터 여부 확인.
sys.stdout.write(f"**소도시 쇼핑액 누락\n{sdf1[sdf1['쇼핑액'].isnull()]}\n")
sys.stdout.write(f"**중도시 쇼핑액 누락\n{sdf2[sdf2['쇼핑액'].isnull()]}\n")
sys.stdout.write(f"**대도시 쇼핑액 누락\n{sdf3[sdf3['쇼핑액'].isnull()]}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


#2 분석 주제에 필요한 '쇼핑액'의 기술 통계량 추출 및 확인
s1 = sdf1.쇼핑액
s2 = sdf2.쇼핑액
s3 = sdf3.쇼핑액

sys.stdout.write(f"@@ 소도시 쇼핑액의 기술통계량\n{s1.describe()}\n")
sys.stdout.write(f"@@ 중도시 쇼핑액의 기술통계량\n{s2.describe()}\n")
sys.stdout.write(f"@@ 대도시 쇼핑액의 기술통계량\n{s3.describe()}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


# 데이터프레임 자료를 넘파이 자료로 변환
snp1 = s1.to_numpy()
snp2 = s2.to_numpy()
snp3 = s3.to_numpy()

# 그룹별 데이터 개수 (행 수)
r1 = len(snp1)
r2 = len(snp2)
r3 = len(snp3)

# 그룹별 평균
avg1 = np.mean(snp1)
avg2 = np.mean(snp2)
avg3 = np.mean(snp3)

# 그룹별 분산
# ddof = 표본 분산 계산 시 자유도를 줄이기 위한 매개변수.
# ddof=1 : 표본의 분산이 모집단 분산을 더 잘 추정할 수 있도록 1을 빼서 보정.
# ddof=0 : 모집단 분산 계산 시 적용.
var1 = np.var(snp1, ddof=1)
var2 = np.var(snp2, ddof=1)
var3 = np.var(snp3, ddof=1)


#3 그룹별 쇼핑액에 대한 분산분석 수행 (f_oneway())
# 그룹 매개변수는 넘파이 배열로 적용한다.
f, pvalue = stats.f_oneway(snp1, snp2, snp3)

sys.stdout.write(f"F통계량 (F-Statistic) = {f}\n")
sys.stdout.write(f"pvalue = {pvalue}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


sys.stdout.write(f"@@ 그룹별 행 개수\n"
                 f"-소도시 = {r1}\n"
                 f"-중도시 = {r2}\n"
                 f"-대도시 = {r3}")
sys.stdout.write("\n-----------------------------------------------------------------\n")
sys.stdout.write(f"@@ 그룹별 평균\n"
                 f"-소도시 = {avg1}\n"
                 f"-중도시 = {avg2}\n"
                 f"-대도시 = {avg3}")
sys.stdout.write("\n-----------------------------------------------------------------\n")
sys.stdout.write(f"@@ 그룹별 분산\n"
                 f"-소도시 = {var1}\n"
                 f"-중도시 = {var2}\n"
                 f"-대도시 = {var3}")
sys.stdout.write("\n-----------------------------------------------------------------\n")



#4 가설 검정
ALPHA = 0.5 #유의수준 5%
if pvalue < ALPHA:
    print("결론: 귀무가설 기각\n"
          "-> 주거지역 그룹 간의 쇼핑액 (분산)차이가 다를 가능성이 있다.")
else:
    print("결론: 귀무가설 채택\n"
          "-> 주거지역 그룹 간의 쇼핑액 (분산)차이가 다를 가능성이 낮다.")