# 단순/다중 선형 회귀 모델 적용 연습

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import re
warnings.filterwarnings('ignore')

# 층('Floor') 설명 문자열을 3개의 분류 기준으로 변경하는 값 처리 함수
def process_special_floors(floor_info):
    match = re.match(r'(\d+) out of (\d+)', floor_info)
    match2 = re.match(r'(\d+)', floor_info)
    if "Ground" in floor_info:
        return "Low"
    elif "Upper Basement" in floor_info:
        return "Low"
    elif "Lower Basement" in floor_info:
        return "Low"
    elif match:
        grp1 = int(match.group(1))
        grp2 = int(match.group(2)) // 2
        if grp1 < grp2:
            return 'Low'
        elif grp1 == grp2:
            return 'Middle'
        else:
            return 'High'
    elif match2:
        return 'Middle'
    else:
        return 'None'


df = pd.read_csv('workpython/rent.csv')
#df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4746 entries, 0 to 4745
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Posted On          4746 non-null   object          > 등록 날짜
 1   BHK                4743 non-null   float64         > 베드, 홀, 키친의 수 합계
 2   Rent               4746 non-null   int64           > 렌트비 (종속변수로 사용)
 3   Size               4741 non-null   float64         > 집 크기
 4   Floor              4746 non-null   object          > 총 층 수의 몇 층
 5   Area Type          4746 non-null   object          > 사이즈 기준, 공용공간 포함 여부
 6   Area Locality      4746 non-null   object          > 지역
 7   City               4746 non-null   object          > 도시
 8   Furnishing Status  4746 non-null   object          > 풀옵션 여부
 9   Tenant Preferred   4746 non-null   object          > 선호하는 가족 형태
 10  Bathroom           4746 non-null   int64           > 화장실 개수
 11  Point of Contact   4746 non-null   object          > 연락할 곳
dtypes: float64(2), int64(2), object(8)
memory usage: 445.1+ KB
"""
half1 = df.iloc[:, 0:6]
#print(half1.head())
"""
    Posted On  BHK   Rent    Size            Floor    Area Type
0  2022-05-18  2.0  10000  1100.0  Ground out of 2   Super Area
1  2022-05-13  2.0  20000   800.0       1 out of 3   Super Area
2  2022-05-16  2.0  17000  1000.0       1 out of 3   Super Area
3  2022-07-04  NaN  10000   800.0       1 out of 2   Super Area
4  2022-05-09  2.0   7500   850.0       1 out of 2  Carpet Area
"""
half2 = df.iloc[:, 6:10]
#print(half2.head())
"""
              Area Locality     City Furnishing Status  Tenant Preferred
0                    Bandel  Kolkata       Unfurnished  Bachelors/Family
1  Phool Bagan, Kankurgachi  Kolkata    Semi-Furnished  Bachelors/Family
2   Salt Lake City Sector 2  Kolkata    Semi-Furnished  Bachelors/Family
3               Dumdum Park  Kolkata       Unfurnished  Bachelors/Family
4             South Dum Dum  Kolkata       Unfurnished         Bachelors
"""
half3 = df.iloc[:, 10:]
#print(half3.head())
"""
   Bathroom Point of Contact
0         2    Contact Owner
1         1    Contact Owner
2         1    Contact Owner
3         1    Contact Owner
4         1    Contact Owner
"""


# DataFrame에서 결측값이 존재하는지 확인
#print(df.isnull().sum())
"""
Posted On            0
BHK                  3
Rent                 0
Size                 5
Floor                0
Area Type            0
Area Locality        0
City                 0
Furnishing Status    0
Tenant Preferred     0
Bathroom             0
Point of Contact     0
dtype: int64

--> 전체 Row 4746 개 중 최소 3개 최대 8개의 Row가 결측값이 있으므로 제거하는 쪽으로 전처리
"""

# 결측값이 존재하는 Row 제거
dfnew = df.dropna()

# 결측값이 제거되면 원본 4746개 중 최소 3개인 4743 또는 최대 8개인 4738개가 남아야 함
#dfnew.info()
"""
<class 'pandas.core.frame.DataFrame'>
Index: 4738 entries, 0 to 4745
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Posted On          4738 non-null   object 
 1   BHK                4738 non-null   float64
 2   Rent               4738 non-null   int64  
 3   Size               4738 non-null   float64
 4   Floor              4738 non-null   object 
 5   Area Type          4738 non-null   object 
 6   Area Locality      4738 non-null   object 
 7   City               4738 non-null   object 
 8   Furnishing Status  4738 non-null   object 
 9   Tenant Preferred   4738 non-null   object 
 10  Bathroom           4738 non-null   int64  
 11  Point of Contact   4738 non-null   object 
dtypes: float64(2), int64(2), object(8)
memory usage: 481.2+ KB
"""


# object 값 중에서 '범주형'에 해당하는 Column 확인.
object_col = dfnew.select_dtypes(include=['object']).columns

for colname in object_col:
    print(f"{colname} Column의 고유값\n"
          f"{dfnew[colname].unique()}")

# 각 Column별 unique 개수 출력
#print(dfnew[object_col].nunique())
"""
Posted On              81
Floor                 480
Area Type               3
Area Locality        2230
City                    6
Furnishing Status       3
Tenant Preferred        3
Point of Contact        3
dtype: int64
"""
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
# Area Locality의 경우 고유값이 너무 많아서 상위 3개만 구분하고 나머지는 '기타' 처리
# 'City'가 대분류로 존재하므로 City별 상위 3개를 구해본다

cities = ['Kolkata', 'Mumbai', 'Bangalore', 'Delhi', 'Chennai', 'Hyderabad']
select_local = []
for city in cities:
    tmp = dfnew[dfnew['City'] == city]['Area Locality'].value_counts()[0:3]
    #print(f"{city}의 주요 3개 지역\n{tmp}")
    for area in tmp.index:
        select_local.append(area)

#print(select_local)

# Area Locality Column에서 select_local 리스트에 있는 것을 제외한 항목명을 'etc'로 변경.
dfnew.loc[:, 'Area Locality'] = dfnew['Area Locality'].apply(lambda x: x if x in select_local else 'etc')
# 'Area Locality' Column에 라벨 인코딩 적용.
dfnew['Area Locality'] = dfnew['Area Locality'].astype('category')

labelencoder = LabelEncoder()
dfnew['Area Locality'] = labelencoder.fit_transform(dfnew['Area Locality'])

#print(dfnew['Area Locality'].head())
"""
0    17
1    17
2    15
4    17
5    17
"""
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------


# 'Posted On' column은 "연-월-일" 값을 갖는다. 이를 분석에 적절히 녹여내기 위해 datetime 객체의 year, month, day_name()을 적용한다.
dfnew['Posted On'] = pd.to_datetime(dfnew['Posted On'])

dfnew['year'] = dfnew['Posted On'].dt.year
dfnew['month'] = dfnew['Posted On'].dt.month
dfnew['weekday'] = dfnew['Posted On'].dt.day_name()     # 일자는 7일제 '요일'로 변경

dfnew['year'] = dfnew['year'].astype('category')
dfnew['month'] = dfnew['month'].astype('category')
dfnew['weekday'] = dfnew['weekday'].astype('category')

#print(dfnew['weekday'].unique())
# ['Wednesday' 'Friday' 'Monday' 'Tuesday' 'Saturday' 'Thursday' 'Sunday']

# 라벨 인코딩 적용
dfnew['year'] = labelencoder.fit_transform(dfnew['year'])
dfnew['month'] = labelencoder.fit_transform(dfnew['month'])
dfnew['weekday'] = labelencoder.fit_transform(dfnew['weekday'])
#print(dfnew['weekday'].head())
"""
0    6
1    0
2    1
4    1
5    0
"""
# Posted On은 필요 없어졌으므로 제거
dfnew = dfnew.drop(columns=['Posted On'], axis=1)
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# 'Floor'의 경우 고유 항목이 너무 많아서 특정 기준으로 3개의 범주형 값으로 변환.
dfnew['floortype'] = dfnew['Floor'].apply(process_special_floors)
dfnew['floortype'] = dfnew['floortype'].astype('category')
#print(dfnew['floortype'].unique())  # ['Low' 'Middle' 'High']
dfnew['floortype'] = labelencoder.fit_transform(dfnew['floortype'])
#print(dfnew['floortype'].unique()) # [1 2 0]

# Floor는 필요 없어졌으므로 제거
dfnew = dfnew.drop(columns=['Floor'], axis=1)
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# 다른 범주형 Column은 항목이 적으므로 바로 라벨 인코딩 적용한다.
col_list = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred']

for col in col_list:
    dfnew[col] = dfnew[col].astype('category')
    dfnew[col] = labelencoder.fit_transform(dfnew[col])

dfnew.info()

# 'Point of Contact'는 분석에 불필요하다 판단해 삭제
dfnew = dfnew.drop(columns=['Point of Contact'], axis=1)
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# 'Rent'를 종속변수 y로 두고, 나머지 Column은 독립변수 x로 활용하기
X = dfnew.drop(columns=['Rent'], axis=1)
y = dfnew['Rent']

# 독립변수로 사용하는 Column 중 수치형(범주X) 데이터는 표준화 작업 수행 (StandardScaler)
scaler = StandardScaler()
col_list = ['BHK', 'Size', 'Bathroom']
X[col_list] = scaler.fit_transform(X[col_list])

# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

# 선형 회귀 모델 테스트를 위한 Train, Test 데이터셋 나누기
trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2024)


# 선형 회귀 모델
model = LinearRegression()
model.fit(trainx, trainy)

rdf = RandomForestRegressor(random_state=2024)
rdf.fit(trainx, trainy)


# 예측
pred = model.predict(testx)
pred2 = rdf.predict(testx)


# 성능 확인
mse = mean_squared_error(testy, pred2)
r2  = r2_score(testy, pred2)

print(mse, r2, model.coef_)

