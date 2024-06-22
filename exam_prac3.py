"""
    3유형 실습 기록


"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols

iris = load_iris()  # 붓꽃 데이터 가져오기
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['species'] = pd.DataFrame(data=iris['target'], columns=['target'])

#print(df['species'], iris['target_names'])
"""
    df['species']는 0, 1, 2 의 값으로 구분되어 있으며
    iris['target_names']는 'setosa, versicolor, virginica'로 구분된다
    이를 replace로 매칭한다.
"""
df['species'].replace([0,1,2], iris['target_names'], inplace=True)

df1 = df[df['species'] == 'setosa']
df2 = df[df['species'] == 'versicolor']

# 2 종류의 petal length 차이가 서로 유의미한지 확인.
# t검정 = scipy의 stats 모듈의 ttest_ind()로 t-검증 수행
# equal_val = False   -->  두 샘플이 동일한 분산을 가지지 않는다고 가정한다. (Welch t-검정 수행)
t, pvalue = stats.ttest_ind(df1['petal length (cm)'], df2['petal length (cm)'], equal_var=False)

print(t, pvalue)

if pvalue < 0.05:
    print("귀무가설 기각")
else:
    print("귀무가설 채택")


#################################################################
#  '비율' 연관성 분석을 위한 카이제곱 검정

pdf = pd.read_csv('./workpython/data.csv', encoding='euc-kr')
df1 = pdf[pdf['주거지역'] == '소도시']  # 소도시 그룹
df2 = pdf[pdf['주거지역'] == '대도시']  # 대도시 그룹

print(df1.info())
print(df2.info())

# 쿠폰 선호도가 있는 사람들 수 계산
dx1 = len(df1[df1['쿠폰선호도'] == '예'])
dx2 = len(df2[df2['쿠폰선호도'] == '예'])
print(dx1, dx2)   # 20 16

print(f"소도시의 쿠폰선호도 '예'인 비율\n"
      f"{dx1/len(df1)}\n"
      f"대도시의 쿠폰선호도 '예'인 비율\n"
      f"{dx2/len(df2)}")

# 카이제곱 검정에 사용할 4개의 변수
# [소도시 쿠폰선호도(예), 대도시 쿠폰선호도(예)], [소도시 쿠폰선호도(아니오), 대도시 쿠폰선호도(아니오)]
observed = [[dx1, dx2], [len(df1)-dx1, len(df2)-dx2]]

chi, pvalue, dof, expect = stats.chi2_contingency(observed)
print(f"{chi}\n{pvalue}\n{dof}\n{expect}")


#############################################################################
# 다중 선형회귀 모형 활용
dff = pd.read_csv('./workpython/mtcars.csv', encoding='euc-kr'
                  , index_col=0)

x = dff[['hp', 'wt', 'am']]  # 3개의 독립변수 활용
y = dff['mpg']  # 종속변수(연비)

fit = ols('y~x', data=dff).fit()   # 다중선형회귀모델 분석
print(fit.summary())

print()
print(f"T-검정통계량 = \n{fit.tvalues}")
print(f"유의수준 p-value = \n{fit.pvalues}")
print()

print(fit.params)
"""
Intercept    34.002875
x[0]         -0.037479      독립1 hp 계수
x[1]         -2.878575      독립2 wt 계수
x[2]          2.083710      독립3 am 계수
"""

#print(dff.iloc[3])
"""
여기서 예측 해보기 위해 샘플로 hp, wt, am 값을 사용한다.
mpg      21.400
cyl       6.000
disp    258.000
hp      110.000   <-
drat      3.080
wt        3.215   <-
qsec     19.440
vs        1.000
am        0.000   <-
gear      3.000
carb      1.000
"""
# 예측하기
pred = fit.predict(exog=dict(x=[[110, 3.215, 0]]))

print()
print(pred[0])  # 20.62559531265174

# 독립변수 중 hp 항목에 대한 95% 신뢰구간
p1 = fit.conf_int(alpha=0.05)[0][1]  # 하한 구간
p2 = fit.conf_int(alpha=0.05)[1][1]  # 상한 구간
print(p1, p2)


########################################################################
# 모의고사 테스트1
data = pd.read_csv('./workpython/recordmath.csv', encoding='euc-kr'
                   , index_col=0)

print(data.columns)
print(data['sex'].unique())
print(data['academy'].unique())

# 학원에 다니는 학생 (academy=1)으로 분류된 남성과 여성의 비율 소수점 2자리까지 계산.
sm = data[(data['academy'] ==1) & (data['sex'] == 'Male')]
sf = data[(data['academy'] ==1) & (data['sex'] == 'Female')]

nm = data[data['sex'] == 'Male']
print(len(sm),  round(len(sm)/len(nm), 2))

nf = data[data['sex'] == 'Female']
print(len(sf), round(len(sf)/len(nf), 2))

# from scipy import stats
# 성별에 따른 academy의 비율에 차이가 있는지 카이제곱 검정통계량 작성. 소수점 3자리까지
# 데이터 행렬 = 남성 학원 O, 여성 학원 O,  남성 학원 X,  여성 학원 X
obs = [[len(sm), len(sf)], [len(nm)-len(sm), len(nf)-len(sf)]]

chi, pvalue, ddf, expect = stats.chi2_contingency(obs)
print(round(chi, 3))


# 유의확률(p-value)을 출력.  유의수준 5% 하에서 가설 검정으 결과 정리.
print(pvalue)
if pvalue < 0.05:
    print("귀무가설 기각")
else:
    print("귀무가설 채택")

