
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

data = pd.read_csv('./workpython/data.csv', encoding='euc-kr',
                   index_col=0)

# csv로 가져온 데이터 확인
print(data.info())
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
None
"""

print(data.isnull().sum())

# 성별이 Object(문자열) Type이므로 구분할 수 있는 수(Float)로 변환하기.
data['gender'] = data['성별'].map({'남자':1, '여자':0})
print(data['gender'].info())

print(data['주거지역'].unique())
data['house'] = data['주거지역'].map({'대도시':2, '중도시':1, '소도시':0})

print(data['쿠폰선호도'].unique())
data['like'] = data['쿠폰선호도'].map({'예':1, '아니오':0})


# 독립변수 구성
x = data[['gender', 'house']]
# 종속변수 구성
y = data['like']


trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3,
                                                random_state=42)

# 나이브베이즈 모델
model = CategoricalNB(alpha=0.8).fit(trainx, trainy)
print(model.get_params())  # 모델 수행 관련 파라미터 값

# 검증 데이터를 이용한 쿠폰선호도 예측.
predictions = model.predict(testx)


print(f"\n@@@@@   범주형 독립변수의 예측 성능   @@@@@")
print(model.score(testx, testy))

print("@@@@@ Confusion Matrix @@@@@")
conf = confusion_matrix(testy, predictions)

print(f"### 분류 분석 모형 성능 평가 지표 ###\n"
      f"{classification_report(testy, predictions)}")

print(f"### F1-Score 계산 모듈 이용 ###\n"
      f"{f1_score(testy, predictions)}")

print(f"### Accuracy 계산 모듈 이용 ###\n"
      f"{accuracy_score(testy, predictions)}")

"""

    유형2는 분류 또는 회귀 모델을 설계하고 예측 모델의 결과를 제출(csv)하는 방식의 문제.
    분류 : 종속(목표)변수 = 범주형 (남/여, 유/무, 가/나 등)
        -> accuracy_score, f1_score, roc_auc_score 등의 확률로 접근 및 해결.
    회귀 : 종속(목표)변수 = 이산형 (가격, 수치 등 )
        -> RMSE, 결정계수 활용

    1. 데이터 로드 및 확인
    2. 결측값 처리 및 라벨인코딩 (문자열 --> 수치형 변환)
    3. 모델링 및 학습
    4. 모델의 성능 평가
    5. 테스트 모델 예측
    6. 테스트 결과 제출 및 확인

"""

