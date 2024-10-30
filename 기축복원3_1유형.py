#제5회 기출 복원 1유형 문제 연습

import pandas as pd
import numpy as np

df = pd.read_csv('./workpython/garbagebag.csv', encoding='euc-kr')
#df.info()

""" 
문제1. 
'종량제봉투처리방식' = 소각용
'종량제봉투사용대상' = 가정용
2L 종량제 봉투의 평균 가격  (단 0인 항목은 제외)
"""
#print(df['종량제봉투처리방식'].unique())
#print(df['종량제봉투사용대상'].unique())

newdf = df[ (df['종량제봉투처리방식'] == '소각용') & (df['종량제봉투사용대상'] == '가정용') ]
newdf = newdf[newdf['2L가격'] != 0]

#print(newdf['2L가격'].mean())
# 95.51162790697674



"""
문제2.
정상인 사람과 과체중인 사람의 차이(명)를 구하기.
BMI = w/(t x t)
세계보건기구 기준 BMI 25 이상인 사람을 과체중으로 분류.
"""
df = pd.read_csv('./workpython/index.csv', encoding='euc-kr')
#df.info()

newdf = df.copy()
newdf['bmi'] = newdf['Weight'] / ((newdf['Height']/100) * (newdf['Height']/100))

#print(newdf['bmi'].head())

overp = newdf[newdf['bmi'] >= 25]
basicp = newdf[newdf['bmi'] < 25]

#print(len(overp), len(basicp), abs(len(overp)-len(basicp)))
# 300


"""
문제3.
순전입학생수(전입학생수합계-전출학생수합계) 구하기.
순전입학생수가 가장 많은 학교의 '순전입학생수'와 '전체학생수'를 출력.
"""
df = pd.read_csv('./workpython/student.csv', encoding='euc-kr')
df.info()

newdf = df.copy()
newdf = newdf.dropna(subset= ['전입학생수합계(명)', '전출학생수합계(명)'])
#newdf.info()
newdf['순전입학생수'] = newdf['전입학생수합계(명)'] - newdf['전출학생수합계(명)']

tmp = newdf.sort_values(['순전입학생수'], ascending=False)
print()

tmp2 = tmp.head(1)

print(tmp2['순전입학생수'])
print()
print(tmp.head(1)['전체학생수합계(명)'])
#print(tmp.iloc[1])


