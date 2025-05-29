
from sklearn.datasets import load_iris              # 데이터셋
from sklearn.metrics import confusion_matrix        # 혼동행렬 구축 모델
from sklearn.metrics import classification_report   # 분류 모형 성능평가 지표
from sklearn.metrics import roc_curve               # 분류 모형 ROC 커브
from sklearn.metrics import auc                     # AUC 값 계산
from sklearn.metrics import f1_score                # F1-Score 계산 모듈
from sklearn.metrics import accuracy_score          # accuracy 계산 모듈
from sklearn.metrics import precision_score         # Precision 계산 모듈
from sklearn.metrics import recall_score            # Recall 계산 모듈

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

iris = load_iris()
x = iris.data
y = iris.target

dfx = pd.DataFrame(x, columns= iris.feature_names)
dfy = pd.DataFrame(y, columns=['species'])
df = pd.concat([dfx, dfy], axis=1)  # 데이터프레임 결합, axis=1 세로축 = 열결합

df = df[['sepal length (cm)', 'species']]   # 꽃받침의 길이, 품종 데이터
df = df[df.species.isin([0,1])]             # 품종(0,1) = (setosa, versicolor) 선택
df = df.rename(columns={'sepal length (cm)': 'sepal_length'})  # Column명 변경

print(" #####   로지스틱 회귀분석 결과 요약  ##### ")
model = sm.Logit.from_formula('species ~ sepal_length', data=df)  # 로지스틱 회귀분석
results = model.fit()  # 모형 적합

print(results.summary())  # 분석 결과 요약 레포트
print(" @@@@@ 판별함수의 값, 첫 10행 출력 @@@@@")
print(results.predict(df.sepal_length)[:10])


print(" @@@@@ 분류 수행 결과(첫 10행), False(0): setosa, True(1): versicolor  @@@@@")
ypred = results.predict(df.sepal_length) >= 0.5  # 판별 기준 0.5
print(ypred[:10])


print(" #####  Confusion Matrix, 혼동행렬  #####")
conf = confusion_matrix(df.species, ypred)
print(conf)

print("  #####  분류 분석 모형 성능평가 지표   #####")
print(classification_report(df.species, ypred))


print("  #####  F1-Score 계산 모듈 이용   #####")
print(f"F1-Score: {f1_score(df.species, ypred)}")


print("  #####  Accuracy(정확도)   #####")
print(accuracy_score(df.species, ypred))


print("  #####  Recall (재현율) 계산   #####")
print(recall_score(df.species, ypred))

print("  #####  ROC Curve   #####")
fpr, tpr, thresholds = roc_curve(df.species, results.predict(df.sepal_length))
print(fpr, tpr)
plt.show()

print(' ***  AUC, Area under ROC Curve, ROC 곡선 아래부분의 면적   ***')
print(auc(fpr, tpr))

