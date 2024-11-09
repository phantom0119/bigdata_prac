# 데이터 전처리 과정 정리

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings(action='ignore')



df = pd.read_csv('workpython/data.csv', encoding='euc-kr')

#print(df.columns)
"""
Index(['성별', '연령대', '직업', '주거지역', '쇼핑액', '이용만족도', '쇼핑1월', '쇼핑2월', '쇼핑3월',
       '쿠폰사용회수', '쿠폰선호도', '품질', '가격', '서비스', '배송', '쇼핑만족도', '소득'],
      dtype='object')
"""

df.info()
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

#---------------------------------------------------------------------------------------

# 쇼핑 1,2,3월과 쇼핑액의 상관관계 확인
newdf = df[['쇼핑1월', '쇼핑2월', '쇼핑3월', '쇼핑액']]
corr_ = newdf.corr(method='pearson')
print(corr_)
"""
          쇼핑1월   쇼핑2월    쇼핑3월      쇼핑액
쇼핑1월  1.000000  0.330779  0.067911  0.742797
쇼핑2월  0.330779  1.000000 -0.035935  0.664616
쇼핑3월  0.067911 -0.035935  1.000000  0.522057
쇼핑액   0.742797  0.664616  0.522057  1.000000
-- 각 월일 쇼핑액과 총 쇼핑액의 상관관계를 비교한 결과,
 1월의 쇼핑액(쇼핑1월)이 쇼핑액에 가장 큰 영향(강한 상관관계-0.7427)을 주고 있는 것으로 확인 가능.
"""

#---------------------------------------------------------------------------------------
# 성별 Column의 데이터 범주화 작업 ( object Type을 'category' Type으로 변환)
print(df['성별'].unique())

## 'gender'라는 새로운 Column에 '성별' 데이터의 category화 된 값을 담기.
df['gender'] = df['성별'].astype('category')
#print(f"({df['gender']}\n\n{df['gender'].dtypes}")


# 범주형 값에서 '남자'를 1,  '여자'를 0으로 매칭하기
#df['gender'] = df['gender'].map({'남자':1, '여자':0})
#print(df['gender'])

# LabelEncoder를 사용해 범주형 데이터를 정규화하기
encoder = LabelEncoder()
nor_gen = encoder.fit_transform(df['gender'])

# 'gender'의 원본 데이터와 변환 데이터를 결합하기
rst = pd.DataFrame({
    'real' : df['gender'],
    'chan' : nor_gen
})

#print(rst)
#---------------------------------------------------------------------------------------
# Object 데이터는 범주화로 변경하고, int와 float 데이터는 StandardScaler()로 표준화하기

newdf2 = df.copy()

# 쇼핑만족도를 종속변수로 두기
y = df['쇼핑만족도']
X = newdf2.drop(['쇼핑만족도', '고객번호'], axis=1)  # 불필요 칼럼과 종속변수 제거

encoder1 = LabelEncoder()
encoder2 = StandardScaler()

contin_var = X.select_dtypes(include=['int64', 'float64']).columns
catego_var = X.select_dtypes(include=['object']).columns
print(contin_var)
print(catego_var)

# 전처리 파이프라인 구축
# LabelEncoder를 ColumnTransformer에 직접 적용할 수 없다.
process = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), contin_var),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), catego_var)
    ]
)
scaled_x = process.fit_transform(X)
# 처리 결과는 ndarray 타입임으로 head() 사용 불가.
print(scaled_x[:10])

X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=52)
model1 = LinearRegression()
model2 = LogisticRegression()
model3 = RandomForestRegressor()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)


# 결정계수(r2) 수준을 보면 LinearRegression 모델의 성능이 더 뛰어난 것을 확인할 수 있다.
print(f"LinearRegression의 r2-score = {r2_score(pred1, y_test)}")       # 1.0
print(f"LogisticRegression의 r2-score = {r2_score(pred2, y_test)}")     # 0.68 수준
print(f"RandomForestRegression의 r2-score = {r2_score(pred3, y_test)}") # 0.99 수준

# LinearRegression의 MSE 값이 월등히 낮은 것으로 우수한 모델임을 확인할 수 있다.
print(mean_squared_error(pred1, y_test))  # 8.326865110666236e-31
print(mean_squared_error(pred2, y_test))  # 0.5555555555555556
print(mean_squared_error(pred3, y_test))  # 0.0003777777777777775



#---------------------------------------------------------------------------------------

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
"""