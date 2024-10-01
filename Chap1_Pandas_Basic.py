# Pandas 기본 활용 연습
import pandas as pd
import numpy as np

df = pd.read_csv('workpython/sale.csv', encoding='euc-kr',
                 header=0, index_col=0 )

#df.info()
#print(df.describe(include='object'))
#print(df.describe())
#print(df.head())
#print(df.tail(4))
#print(df.columns)
#print(df.index)
#print(df.dtypes)
#print(df.isnull().sum())
#print(df.nunique())

"""
df_new = df.select_dtypes(include=['float', 'int'])
print(df_new.dtypes)
# 상관계수 계산은 수치형 데이터끼리 가능하다.
print(df_new.corr())
"""
#print(df['주구매상품'].value_counts())

"""
# 임시 데이터 프레임 만들기
data = [1,1,2,3,4,5,6,6,6,7,8,9,9]
dtmp = pd.DataFrame(data, columns=['value'])

print(f"사분위 수 0% : {dtmp['value'].quantile(0)}")
print(f"min() 결과: {dtmp['value'].min()}")
print(f"사분위 수 하위 25% : {dtmp['value'].quantile(0.25)}")
print(f"median() 결과: {dtmp['value'].median()}")
print(f"사분위 수 하위 50% : {dtmp['value'].quantile(0.5)}")
print(f"사분위 수 하위 75% (상위 25%) : {dtmp['value'].quantile(0.75)}")
print(f"max() 결과: {dtmp['value'].max()}")
print(f"사분위 수 하위 100% : {dtmp['value'].quantile(1.0)}")
"""


# 데이터 정렬과 인덱싱
# print(f"주구매지점 정렬 전 0~4 인덱스 데이터\n"
#       f"{df.iloc[0:5, [0,1,3,4]]}")
#print()
#print(df.loc[0:4, ['총구매액', '최대구매액', '주구매상품', '주구매지점']])



## 인덱스 슬라이싱이 불가능한 경우 (인덱스가 정수 값 구조가 아닐 때)
data = [['S', 'VVIP_room', 200000],
        ['A', 'VIP_room', 150000],
        ['B', 'Business', 80000],
        ['C', 'Multi_room', 50000],
        ['D', 'Basic', 35000]]

tmpdf = pd.DataFrame(data, columns=['RoomClass', 'Name', 'Price'],
                     index=['S', 'A', 'B', 'C', 'D'])
#print(tmpdf.loc['S', ['Name', 'Price']])



## 특정 행만 추출하기 위한 조건을 적용할 때는 loc를 사용한다.
printdf = df.loc[df['총구매액'] >= 30000000, ['총구매액', '주구매상품', '주구매지점']]
#print(printdf[0:5])

"""
# 오름차순으로 확인
tmpdf = df.sort_values(['주구매지점'], ascending=True, inplace=False)
print(tmpdf.iloc[0:4, [3, 4, 5, 6]])

print(tmpdf.loc[0:4, ['주구매상품', '주구매지점', '내점일수', '내점당구매건수']])
"""



nadf = df[df['환불금액'].isnull() == True]
nodf = df.dropna(axis=0, subset=['환불금액'], inplace=False)

nodf.info()








