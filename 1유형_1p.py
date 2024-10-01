# 빅데이터분석기사 실기 1유형 문제 접근 및 코드 작성 정리


import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys


#seaborn에 있는 데이터셋 리스트 확인
idx=1
for item in sns.get_dataset_names():
    if idx%6 == 0:
        print()
    sys.stdout.write(f"{item}, ")
    idx+=1
print("\n-------------------------------------------------------------")

df = sns.load_dataset('tips')
print(df.head(15))
print("\n-------------------------------------------------------------")
print(df.info())
print("\n-------------------------------------------------------------")



"""
Q1. total_bill  변수의 제 1사분위 수 구하고 정수로 출력
"""
print("total_bill  변수의 제 1사분위 수 구하고 정수로 출력")
print(f"{df['total_bill'].head(15)}")

#사분위수 Function = 데이터프레임에서 quantile()
# 파라미터 = 분위수 (0.25 = 1사분,  0.5 = 2사분(중앙값),  0.75 = 3사분,  1.0 = 4사분
print(f"1사분면 = {df['total_bill'].quantile(0.25)}")
print(f"1사분면 정수화 = {int(df['total_bill'].quantile(0.25))}")

print(f"2사분면 = {df['total_bill'].quantile(0.5)}")
print(f"3사분면 = {df['total_bill'].quantile(0.75)}")
print(f"4사분면 = {df['total_bill'].quantile(1.0)}")

quant3 = df['total_bill'].quantile(0.75)
quant1 = df['total_bill'].quantile(0.25)
print(f"IQR값? = (3사분 - 1사분) : {quant3-quant1}")
print("\n-------------------------------------------------------------")



"""
Q2. total_bill 값이 20이상 30이하의 데이터 수 구하기.
"""
print("total_bill 값이 20이상 30이하의 데이터 수 구하기.")
q2_df = df[(df['total_bill']>=20) & (df['total_bill'] <=30)]

print(len(q2_df['total_bill'].to_numpy()))

cond1 = (df['total_bill'] >= 20)
cond2 = (df['total_bill'] <= 30)
q2_df2 = df[cond1 & cond2]
print(len(q2_df2))
print("\n-------------------------------------------------------------")


"""
Q3. tip변수의 상위 10개 값 추출  -->  총합  --> 소수점을 버려서 정수로 출력
"""
print("Q3-1. tip변수의 상위 10개 값 가져오기")
tip10 = df['tip'].sort_values(ascending=False)[0:10]
print(tip10)

print(f"Q3-2. 상위 10개의 총합 = {tip10.sum()}")
print(f"Q3-3. 소수점 버려서 정수 출력 = {int(tip10.sum())}")
print("\n-------------------------------------------------------------")



"""
Q4. sex Column에서 'Female' 비율을 소수점 2자리까지 추출
"""
print("전체 데이터 중 여성의 비율")
print(f"{df['sex'].unique()}")

Fe = df[df['sex'] == 'Female']
print(f"{len(Fe)/len(df)}")
print(f"{round(len(Fe)/len(df), 2)}")
print("\n-------------------------------------------------------------")



"""
Q5. DataFrame 자료를 순서대로 10개 추출   -> total_bill의 열 평균값을 반올림 후 소수점 1자리까지 추출
"""
print(f"DataFrame 자료의 첫 10개 행 자료")
print(df.head(10))
df_new = df.head(10)

total_bill_col = df_new['total_bill'].mean()
print(f"total_bill 평균: {total_bill_col}, 소수점 1자리 반올림: {round(total_bill_col, 1)}")
print("\n-------------------------------------------------------------")



"""
Q6. time이 Lunch, Dinner인 각 경우의 total_bill의 평균과 표준편차, 분산 구하기.
    그리고 각 경우의 tip 변수에 대해 최소-최대 척도(Min-Max Scale)와 Z-score 구하기.
"""
print(f"time 변수의 도메인 확인 = {df['time'].unique()}")

df_lun = df[df['time'] == 'Lunch']
df_din = df[df['time'] == 'Dinner']

print(f"Lunch의 total_bill 평균 = {df_lun['total_bill'].mean()}")
print(f"Dinner의 total_bill 평균 = {df_din['total_bill'].mean()}")
print(f"Lunch의 total_bill 표준편차 = {df_lun['total_bill'].std()}")
print(f"Dinner의 total_bill 표준편차 = {df_lun['total_bill'].std()}")
print(f"Lunch의 total_bill 분산 = {df_lun['total_bill'].var()}")
print(f"Dinner의 total_bill 분산 = {df_lun['total_bill'].var()}")
print("\n-------------------------------------------------------------")


"""
Scaler는 2차원 데이터프레임 자료를 요구하기 때문에 Column을 1개만 적용하는 경우에는 2차원 형식으로 맞춰줘야 한다.
"""
min_max_scaler = MinMaxScaler()
mms_lun = min_max_scaler.fit_transform(df_lun[['tip']])       # 라이브러리를 이용한 Min-Max Scaler 계산

"""
최소-최대 척도 계산
-> ( X값 - X최소값 ) / ( X최대값 - X최소값 )
"""
lun_tip = df_lun['tip']
mms_lun2 = (lun_tip - lun_tip.min()) / (lun_tip.max() - lun_tip.min())

#Min-Max 척도 비교
print(f"MinMaxScaler 객체 적용 결과\n{mms_lun[0:5]}")
print('**********************************')
print(f"Min-Max Scaler 수동 계산 결과\n{mms_lun2[0:5]}")
print("\n-------------------------------------------------------------")



std_scaler = StandardScaler()
std_din = std_scaler.fit_transform(df_din[['tip']])

"""
Z-Score  (StandardScaler) 계산식
-> ( X값 - X평균 ) / X표준편차
"""

din_tip = df_din['tip']
std_din2 = ( din_tip - din_tip.mean()) / din_tip.std()


# StandardScaler 척도 비교
print(f"StandardScaler 객체 적용 결과\n{std_din[0:5]}")
print('**********************************')
print(f"Z-Score 수동 계산 결과\n{std_din2[0:5]}")
print("\n-------------------------------------------------------------")


# 표준화 데이터셋의 결과는 평균이 0에 가깝고, 표준편차가 1로 고정된다.
print(f"Z-score 평균: {std_din.mean()}")
print(f"Z-score 표쥰편차: {std_din.std()}")
print("\n-------------------------------------------------------------")





"""
Q7. size 변수를 기준으로 중앙 값을 구하고, 중앙값 이상이면 '1', 중앙값 미만이면 '0'으로 분류하는
    새로운 컬럼 'classify'를 추가한다.
"""
print(f"size 변수의 도메인: {df['size'].unique()}")

#중앙값 구하기
med = df['size'].median()
print(med)

df_new = df.copy()
df_new['classify'] = np.where(df_new['size'] >= med, 1, 0)

print(f"중앙값 이상인 size 개체 수: {len(df_new[df_new['classify'] == 1])}")
print(f"중앙값 미만인 size 개체 수: {len(df_new[df_new['classify'] == 0])}")










