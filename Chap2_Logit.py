
"""
    로지스틱 회귀 분류
      ★ 종속 변수가 범주형 ([남, 녀], [A, AB, B, O])자료일 때  사용.

"""

from sklearn.linear_model import LogisticRegression  #로지스틱 회귀분석
from sklearn.preprocessing import StandardScaler     #데이터 전처리(표준화)
from sklearn.datasets import make_classification     #랜덤 데이터 생성
from sklearn.metrics import confusion_matrix         #혼동행렬
from sklearn.metrics import roc_curve                # ROC 그래프

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


iris = load_iris()  # iris 데이터셋 사용
X = iris.data

#(iris.feature_names)   # 4개 특성 (독립변수)
#print(iris.target_names)    # Class ( 3가지 품종 )

scaler = StandardScaler()   #데이터 표준화 작업
X_scaled = scaler.fit_transform(X)
#print(X_scaled[:5])
#(X[:5])

"""
    np.c_  :  여러 배열을 열(Column) 기준으로 병합할 때 사용.
      따라서 np.c_[iris['data'], iris['target'] : 'data' 와 'target' Column 결합.
    -> 기존의 4개의 열에서 'target' 열이 추가되는 것.
    
    columns 구조도 맞춰야 하므로 columns 항목에 'target' 추가
"""
irisdata = pd.DataFrame(data=np.c_[iris['data'], iris['target']]
                         , columns =iris['feature_names'] + ['target'])

#0 ~ 2로 기록된 클래스 번호
#print(irisdata['target'])

irisdata['target'] = irisdata['target'].map({0: 'setosa'
                                        ,   1: 'versicolor'
                                        ,   2: 'virginica'})
# 각 클래스 번호별로 매핑된 결과값 -> 특징 분류 이름.
#print(irisdata['target'])


print(irisdata.dtypes)  #데이터셋의 각 Column Type 확인.
print(irisdata.head(4)) #첫 4개의 Row 확인.
print(irisdata.columns) #데이터셋의 columns list 확인.

# 독립변수 4개 세팅.
features = irisdata[['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)' ]]

labels = irisdata['target']     # 종속변수 세팅
scaler = StandardScaler()       # Z-score 표준화
scaler.fit(features)            # 독립변수 표준화 계산 결과.
x = scaler.transform(features)  # 독립변수 표준화 값 변환.

print("############# 표준화 전 특징값(독립변수) 분포 #############")
print(features[:10])
print("############# Z-score 표준화 후 특징값(독립변수) 분포 #############")
print(x[:10])


# C: 정규화 강도 조절. 값이 클수록 정규화가 약해짐  --> 모형이 학습 데이터에 더 많이 적합하려고 시도.
# C 값이 작을수록 정규화가 강화되어 모형이 간단하게 유지되려고 시도한다.
# ★★ C=20  :  "릿지 회귀"  "L2 규제"
# max_iter  :  최대 반복 학습과정 횟수 (기본 100)
# 로지스틱회귀는 "경사하강법"을 적용한 학습.
model = LogisticRegression(C=20, max_iter=1000)

model.fit(x, labels)  # 표준화된 x값과 종속변수를 매칭 후 학습
print(f"로지스틱 회귀모형 정확도 (%) :"
      f" {model.score(x, labels)*100}")

print(f" ****  분류 유형  ****")
print(model.classes_)

print(f"계수의 크기(특성 개수 4개) : {model.coef_.shape}")
print(model.coef_)

print(f"z절편값(분류 클래스의 개수=3)의 크기: "
    f"{model.intercept_.shape}")
print(model.intercept_)

