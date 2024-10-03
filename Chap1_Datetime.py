# Pandas Datetime Type 활용  실습

import pandas as pd
import numpy as np
import datetime as dt

df = pd.read_csv('workpython/air_pollution_data.csv', encoding='euc-kr'
                 , header=0, index_col=0)

print(df.head())

#1 Datetime 타입의 새 Column 구성하기.
df['측정연월일'] = df['dataTime'].apply(lambda row: pd.to_datetime(str(row), format='%Y-%m-%d'))

print(df.head())
print(df['측정연월일'].dtypes)

# datetime 라이브러리를 이용하기 위해
# import datetime as dt  추가
df['연도'] = df['측정연월일'].dt.year
df['월'] = df['측정연월일'].dt.month
df['일'] = df['측정연월일'].dt.day

tmpdf = df[['dataTime', '측정연월일', '연도', '월', '일']]
print(tmpdf.head())
print(tmpdf.dtypes)
print("-----------------------------------------------------------------")


# datetime 객체 다루기
time = dt.datetime.now()
print(f"현재 시간: {time}")
print("-----------------------------------------------------------------")

print(f"연도: {time.year}")
print(f"월: {time.month}")
print(f"일: {time.day}")
print(f"시: {time.hour}")
print(f"분: {time.minute}")
print(f"초: {time.second}")
print(f"마이크로초: {time.microsecond}")
print(f"요일 반환 (월:0, 화:1, 수:2 ... 일:6): {time.weekday}")
print(f"문자열로 반환: {time.strftime}")

# 특정 형식으로 문자열 변환
"""
%A: 요일 (Monday, Tuesday, Wednesday, Thursday...)
%B: 월   (January, February, March, April...
%Y: 4자리 연도 (2024)
%y: 2자리 연도 (24)
%m: 2자리 월   (01, 02...12)
%d: 2자리 일   (01, 02...31)
%H: 24시간 형식 시 (00, 01...24)
%M: 2자리 분   (00, 01...59)
%S: 2자리 초   (00, 01...59)

%I: 12시간 형식 시 (01, 02...12)
%p: 오전/오후  (AM/PM)
"""
print("-----------------------------------------------------------------")
print(f"datetime을 문자열 구조로 변환= {time.strftime("%p %I:%M:%S")}")
text_time = time.strftime("%p %I:%M:%S")
print(f"text_time 변수의 타입 = {type(text_time)}")

print(f"문자열을 datetime 구조로 변환= {dt.datetime.strptime(text_time, '%p %I:%M:%S')}")
changed_time = dt.datetime.strptime(text_time, '%p %I:%M:%S')
print(f"{type(changed_time)}")


print(f"날짜만 출력하는 date class: {time.date()}")
print(f"시간만 출력하는 time class: {time.time()}")

