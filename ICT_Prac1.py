import pandas as pd
import numpy as np

route = "E:\\수도권ICT교육\\6월\\2024-06-10\\Pandas 실습용 데이터\\seoul_house_price.csv"

df = pd.read_csv(route)
print("############# Data Sample #############")
print(df.head())
print("############# Data Info   #############")
print(df.info())
print("############# Data Null   #############")
print(df.isnull().sum())

# Column명 변경하기
df.rename(columns={'분양가격(㎡)':'분양가격'}, inplace=True)
print(df.head())


# 통계 확인하기
# 자료에서 연도와 월 데이터만 int64 Type이므로 2개의 통계가 나타난다.
print(df.describe())

# 정수로 바꾸고 싶은 Column 벡터 변경.
# 주의 : 분양가격은 None 값이 295개 존재.
#df['분양가격'].astype(int)
# --> invalid literal for int() with base 10: '  '

space_mask = df['분양가격'] == '  '   # 조건에 매칭되는 마스크 생성

# 공백만 존재하는 위치의 값을 NaN으로 변경
df.loc[df['분양가격'] == '  ', '분양가격'] = np.nan
print(df.isnull().sum())

"""
    결측값 삭제해도 되는 경우 : 기준을 구분 할 필요가 없는 경우, 데이터의 양이 많은 경우.
    결측값 삭제가 손해인 경우 : 다른 Column 구성이 기준이 될 수 있는 경우 (고유 데이터).
"""
df['분양가격'] = df['분양가격'].fillna(0)
#df['분양가격'].astype(int)
# --> invalid literal for int() with base 10: '6,657'

# 쉼표(,) 문자를 지우기
df['분양가격'] = df['분양가격'].str.replace(',', '')
#df['분양가격'].astype(int)
# --> cannot convert float NaN to integer

df['분양가격'] = df['분양가격'].fillna(0)
# df['분양가격'].astype(int)
# --> invalid literal for int() with base 10: '-'


df.loc[df['분양가격'] == '-', '분양가격'] = 0
df['분양가격'] = df['분양가격'].astype(int)
print(df.info())
print(df.describe())
print(df.isnull().sum())