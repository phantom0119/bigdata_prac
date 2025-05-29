

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.DataFrame(iris['target'], columns=['target'])
df['species'] = df['species'].replace([0,1,2], iris['target_names'])

#print(df['species'].unique())

setodf = df[df['species'] == 'setosa']
versidf = df[df['species'] == 'versicolor']

print(setodf.head())
print(versidf.head())

model = stats.ttest_ind(setodf['petal length (cm)'], versidf['petal length (cm)'], equal_var=False)
print(model)



df = pd.read_csv('workpython/data.csv', encoding='euc-kr')
print(df.head())

sodf = df[df['주거지역'] == '소도시']
daedf = df[df['주거지역'] == '중도시']

sod1 = sodf[sodf['쿠폰선호도'] == '예']
dae1 = daedf[daedf['쿠폰선호도'] == '예']
sodlen = len(sodf)
daelen = len(daedf)

#print(len(sod1), sodlen)

# 카이제곱-contingency 검정을 위한 테이블 생성
"""
구조
                    소도시           중도시
쿠폰선호도[예]      len(sod1)        len(dae1)
쿠폰선호도[아니오]  len()-sod1      len()-dae1
"""

table = [[len(sod1), len(dae1)], [sodlen-len(sod1), daelen-len(dae1)]]

model = chi2_contingency(table)
print(model)



dfnew = df[['품질', '가격', '서비스', '배송']]
print(dfnew.head())

# Row 단위의 데이터 구성을 위해 Group Column 추가하기
dfnew['s1'] = 1  # 품질 그룹
dfnew['s2'] = 2  # 가격 그룹
dfnew['s3'] = 3  # 서비스 그룹
dfnew['s4'] = 4  # 배송 그룹

d1 = dfnew[['품질', 's1']].to_numpy()
d2 = dfnew[['가격', 's2']].to_numpy()
d3 = dfnew[['서비스', 's3']].to_numpy()
d4 = dfnew[['배송', 's4']].to_numpy()

dataset = pd.DataFrame(np.concatenate((d1,d2,d3,d4), axis=0),
                       columns=['value', 'group'])

print(dataset)

# group별 데이터 간의 차이를 분석해 그룹 간의 평균 차이가 통계적으로 유의미한지 평가한다.
model = ols('value ~ C(group)', data=dataset).fit()











