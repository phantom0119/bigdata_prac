# 빅데이터 분석기사 실기 1유형 연습


import pandas as pd
import numpy as np

df = pd.read_csv('workpython/airquality.csv', encoding='utf-8',
                 index_col=0)

print(f"데이터프레임 컬럼 구성\n{df.columns}\n")
print(f"데이터프레임 컬럼 정보\n{df.dtypes}")
print("##################################################")

"""
Q1. 1. 태양 복사량(Solar.R)을 내림차순으로 정렬.
    2. 전체 자료 중 80%의 자료를 저장 (랜덤 추출 허용) --> data 변수
    3. data 변수를 이용해 Ozone 항목의 결측값을 Ozone 항목의 평균값으로 대체.
    4. Ozone 항목에 대해 (평균값 대체 전 중앙값) - (평균값 대체 후 중앙값) 계산.
"""

#1 내림차순 정렬
ndf = df.sort_values(by='Solar.R', ascending=False)

print(f"정렬 전 df\n{df.head(5)}\n"
      f"정렬 후 df\n{ndf.head(5)}")
print("##################################################")


#2 전체 자료의 80%만 data 변수에 담기  (랜덤 추출 허용)
cnt = len(ndf) * 0.8
print(cnt)

# reset_index() : 데이터프레임의 인덱스를 재구성
#  --> drop 옵션: 기존의 인덱스를 제거할 지 선택 (True = 제거,  False = 이전 인덱스를 새로운 열에 추가)
ndf80 = ndf.sample(frac=0.8, random_state=5538).reset_index(drop=True)

print(f"수집한 80% 수준의 샘플 개수 : {len(ndf80)}")
print("##################################################")




#3 'Ozone' 변수의 결측값을 'Oznoe' 변수의 평균값으로 대체.
mean_value = ndf80['Ozone'].mean()
print(f"Ozone 항목의 평균: {mean_value}")
median_set = ndf80['Ozone'].fillna(mean_value)
print("##################################################")


#4 Ozone 항목에 대해 (평균값 대체 전 중앙값) - (평균값 대체 후 중앙값) 계산.
median_na   = ndf80['Ozone'].median()
median_fill = median_set.median()

print(median_na - median_fill)
print("##################################################")




"""
Q2.  1. 결측값이 제거된 Ozone 항목을 이용해 사분위수를 구한다.
     2. 상위 25% 이상, 하위 25% 이하의 값을 모두 0으로 대체한다.
     3. 대체된 셋을 이용해 Ozone에 대한 평균과 표준편차를 구한다.
"""

ndf = df.dropna(subset=['Ozone'])
#print(len(ndf[ndf['Ozone'].isnull()]))

print(f"Ozone의 결측값을 제거한 사분위 수 계산\n"
      f"1사분위: {ndf['Ozone'].quantile(0.25)}\n"
      f"2사분위: {ndf['Ozone'].quantile(0.5)}\n"
      f"3사분위: {ndf['Ozone'].quantile(0.75)}\n"
      f"4사분위: {ndf['Ozone'].quantile(1.0)}\n")
print("##################################################")



#2 상위 25% 이상,  하위 25% 이하의 값을 0으로 대체.
#   -- 상위 25% 이상 : 3분위수 이상의 값
#   -- 하위 25% 이하 : 1분위수 이하의 값
q1 = ndf['Ozone'].quantile(0.25)
q3 = ndf['Ozone'].quantile(0.75)

ndf.loc[ndf['Ozone'] >= q3, 'Ozone'] = 0
ndf.loc[ndf['Ozone'] <= q1, 'Ozone'] = 0

print(ndf['Ozone'])



#3 변환 후의 Ozone 항목의 평균과 표준편차 계산
mean_v = ndf['Ozone'].mean()
std_v = ndf['Ozone'].std()

print(mean_v, std_v)




"""
실기 도서 1유형 문제 1.
 mtcars.csv
"""
df = pd.read_csv('workpython/mtcars.csv')
df = df.sort_values(by=['mpg'], ascending=False)
dfnew = df.head(10)

mean_carb2 = dfnew[dfnew['carb'] == 2].hp.mean()
mean_carb1 = dfnew[dfnew['carb'] == 1].hp.mean()

#print(mean_carb1, mean_carb2, mean_carb2-mean_carb1)
# 77.4 82.6 5.199999999999989

"""
실기 도서 1유형 문제 2.
mtcars.csv
"""
df = pd.read_csv('workpython/mtcars.csv')
newdf = df[df['am'] == 1]
newdf_4 = newdf[newdf['cyl'] == 4]
#print(newdf_4.mpg.mean(), newdf_4.hp.std(), newdf_4.mpg.mean()+newdf_4.hp.std())
# 28.075 22.65541562503008 50.730415625030076


"""
실기 도서 1유형 문제 3.
mtcars.csv
"""
newdf = df[df['am'] == 0]
iqr = newdf['mpg'].quantile(0.75) - newdf['mpg'].quantile(0.25)
meanv = round(newdf['mpg'].mean(), 3)

rstdf = newdf[(newdf['mpg'] <= (meanv-iqr)) | (newdf['mpg'] >= (meanv+iqr))]
m = rstdf.mpg.mean()
cnt = rstdf.shape[0]

#print(iqr, meanv)
#print(rstdf.head(), m, cnt)
# 18.483333333333334 6



"""
실기 도서 1유형 문제 1.
 sklearn.datasets import load_iris
"""
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['target'])

print(df.columns)
t = df['petal length (cm)'].mean() + df['petal length (cm)'].std()
#print(t)
# 5.5232982332594664

#print(df['target'].unique())

seto = df[df['target'] == 0]
#print(seto.head())

# min-max 수동식
base = seto['petal length (cm)']
#seto['minmax'] = (base-base.min(axis=0)) / (base.max(axis=0)-base.min(axis=0))

#print(seto['minmax'].head())




"""
실기 도서 1유형 문제 2.
sklearn.datasets import load_iris
"""
newdf = df[(df['target']==1) | (df['target']==2)]
newdf = newdf.head(100)

versi = newdf[newdf['target']==1]
virgi = newdf[newdf['target']==2]

versi_mid = versi['sepal length (cm)'].median()
virgi_mid = virgi['petal length (cm)'].median()

ve_cnt = versi[versi['sepal length (cm)'] >= versi_mid].shape[0]
vi_cnt = virgi[virgi['petal length (cm)'] >= virgi_mid].shape[0]

print(ve_cnt, vi_cnt, ve_cnt/versi.shape[0], vi_cnt/virgi.shape[0])











