# 데이터 전처리 과정 정리

import pandas as pd


df = pd.read_csv('workpython/data.csv', encoding='euc-kr',
                 index_col=0)

#print(df.columns)
"""
Index(['성별', '연령대', '직업', '주거지역', '쇼핑액', '이용만족도', '쇼핑1월', '쇼핑2월', '쇼핑3월',
       '쿠폰사용회수', '쿠폰선호도', '품질', '가격', '서비스', '배송', '쇼핑만족도', '소득'],
      dtype='object')
"""

#df.info()
"""
<class 'pandas.core.frame.DataFrame'>
Index: 90 entries, 190105 to 190194
Data columns (total 17 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   성별      90 non-null     object 
 1   연령대     90 non-null     object 
 2   직업      90 non-null     object 
 3   주거지역    90 non-null     object 
 4   쇼핑액     90 non-null     float64
 5   이용만족도   90 non-null     int64  
 6   쇼핑1월    90 non-null     float64
 7   쇼핑2월    90 non-null     float64
 8   쇼핑3월    90 non-null     float64
 9   쿠폰사용회수  90 non-null     int64  
 10  쿠폰선호도   90 non-null     object 
 11  품질      90 non-null     int64  
 12  가격      90 non-null     int64  
 13  서비스     90 non-null     int64  
 14  배송      90 non-null     int64  
 15  쇼핑만족도   90 non-null     int64  
 16  소득      90 non-null     int64  
dtypes: float64(4), int64(8), object(5)
memory usage: 12.7+ KB
"""


"""
# 성별 Column의 데이터 범주화 작업 ( object Type을 'category' Type으로 변환)
print(df['성별'].unique())

## 'gender'라는 새로운 Column에 '성별' 데이터의 category화 된 값을 담기.
df['gender'] = df['성별'].astype('category')

print(f"({df['gender']}\n\n{df['gender'].dtypes}")


# 범주형 값에서 '남자'를 1,  '여자'를 0으로 매칭하기
df['gender'] = df['gender'].map({'남자':1, '여자':0})
print(df['gender'])
"""


"""
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [3, 4, 6], 'C': ['대전', '대구', '광주']})
df2 = pd.DataFrame({'A': [4, 5, 6], 'B': [2, 9, 5], 'C': ['인천', '울산', '부산']})

# 행 기준 연결 (axis=0)
result = pd.concat([df1, df2], axis=0)
result = result.reset_index(drop=True)
#print(result)


df3 = pd.DataFrame({'A': [1, 2, 3], 'D': [3, 4, 6], 'C': ['대전', '대구', '광주']})
df4 = pd.DataFrame({'B': [2, 5, 7], 'E': ['남자', '남자', '여자']})

# 열 기준 연결 (axis=1)
rst = pd.concat([df3, df4], axis=1)
print(rst)


# 인덱스 재구성
result = result.reset_index(drop=False)
print(result)
"""



# merge() 함수 사용 실습
df1 = pd.DataFrame({'key_left': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key_right': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

rst = pd.merge(df1, df2, left_on='key_left', right_on='key_right', how='outer')
# print(rst)
# print()

df3 = pd.DataFrame({'key': ['Dog', 'Cat', 'Fish'], 'value1': [20, 15, 5]})
df4 = pd.DataFrame({'key': ['Cat', 'Dog', 'Deer'], 'value2': ['Open', 'Close', 'Close']})

rst = pd.merge(df3, df4, on='key', how='outer', sort=False)
print(rst)