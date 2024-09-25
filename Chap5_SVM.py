#서포트 벡터 머신 (SVM)

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score

df = pd.read_csv('workpython/train_commerce.csv', encoding='utf-8', index_col=0)
sys.stdout.write(f"데이터프레임 구성 컬럼\n{df.columns}")
sys.stdout.write("\n-----------------------------------------------------------------\n")

# 결측값 제거
df = df.dropna()

# 데이터 유형 확인
sys.stdout.write(f"데이터 유형\n{df.dtypes}")
sys.stdout.write("\n-----------------------------------------------------------------\n")

# 데이터 구성 확인
sys.stdout.write(f"Customer_care_calls 데이터 구성 확인\n"
                 f"{df['Customer_care_calls'].unique()}\n")

sys.stdout.write(f"Prior_purchases 데이터 구성 확인\n"
                 f"{df['Prior_purchases'].unique()}\n")

sys.stdout.write(f"Product_importance 데이터 구성 확인\n"
                 f"{df['Product_importance'].unique()}\n")

# 독립변수 구성 중 'Product_importance'의 데이터는 문자열(object)임을 확인.
# 정수 형태의 범주형 값으로 변경 작업 진행

df['Product_importance'] = df['Product_importance'].map({'low':0, 'medium':1, 'high':2})
sys.stdout.write("\n-----------------------------------------------------------------\n")

x = df[['Customer_care_calls', 'Prior_purchases', 'Product_importance']].to_numpy()     # 독립변수 Set
y = df['Reached.on.Time_Y.N'].to_numpy()                                                # 종속변수 y
#print(x)
"""
[[4 3 0]
 [4 2 0]
 [2 4 0]
 ...
 [5 5 0]
 [5 6 1]
 [2 5 0]]

"""

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=5538)

"""
StandardScaler = 표준화 객체
표준화 z = (x - μ) / σ
 - x : 데이터 값
 - μ : x 집합의 평균
 - σ : 표준 편차
 
 데이터 표준화 작업을 거치면 x 집합의 분포가 평균이 0, 표준편차가 1이 되도록 구성된다.
 이는 학습 시 정교한 결과를 반환하여 성능 상의 이점이 된다.
 
"""
scaler = StandardScaler()
trainx_scaler = scaler.fit_transform(trainx)
testx_scaler = scaler.fit_transform(testx)

# print(trainx_scaler)
"""
[[-1.80602101  0.29400458 -0.94427666]
 [ 2.58221221  0.29400458 -0.94427666]
 [-0.92837437  1.61965099 -0.94427666]
 ...
 [ 0.82691892  1.61965099 -0.94427666]
 [-0.92837437 -1.03164183 -0.94427666]
 [ 0.82691892  1.61965099  0.61279895]]
"""

# SVM 모델 생성
model = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
"""
SVC : 서포트 벡터 머신 분류 모델 (2개 이상의 Class 분류 시 사용)
 - kernel 옵션 : 분류를 위한 변환 방식 (커널 함수) 설정
    ▶ rbf : Radial Basis Function (방사 기저 함수) = 비선형 데이터 분류에 적용. 
    ▶ linear : 데이터가 선형적으로 분리 가능 시 적용하는 커널.
    ▶ poly : 다항식 형태로 데이터 변환 후 분류. degree 옵션으로 차수 설정.
    ▶ sigmoid : 신경망의 시그모이드 함수와 유사하게 데이터 변환 (비선형 데이터 분류)
    ▶ precomputed : 미리 계산된 커널 매트릭스를 입력으로 사용 시 적용.
    
- gamma 옵션 : rbf kernel 시 모델이 얼마나 영향을 미치는지 결정하는 '감마 값'
  > 값이 크면 Data Point가 자신에게 가까운 Point에만 영향을 미침.
  > 값이 작으면 멀리 있는 Point에도 영향을 미침.
  > 0.1 수준은 넓은 영역의 데이터 포인트에 영향을 미친다는 설정.    

- C 옵션 : 정규화 Parameter (규제 수준)
  > 오차를 어느 정도의 수준까지 허용할지 지정.
  > 값이 클수록 모델은 오차를 더 적게 허용.
  > 값이 작을수록 모델은 오차를 더 많이 허용.
  > 10은 값이 작은 수준이며 이는 학습 데이터에 맞추려는 성향이 강해진단 의미.
  > 값이 커지면 과적합(Overfitting) 주의

"""
model.fit(trainx_scaler, trainy)
pred_y = model.predict(testx_scaler)


# 모델 성능 정확도 (Accuracy)
acc = accuracy_score(pred_y, testy)  # 매개변수 순서는 상관 없음
sys.stdout.write(f"모델 정확도 = {acc}\n")

conf_metrix = confusion_matrix(testy, pred_y)
sys.stdout.write(f"혼동 행렬\n{conf_metrix}\n")
"""
[[  21 1272]        True Negative(TN)   |  False Positive(FP)
 [  40 1967]]       False Negative(FN)  |  True Positive(TP)

 혼동행렬 구조
                    예측 Negative(0)     |     예측 Positive (1)
 실제 Negative (0)        TN             |          FP
 실제 Positive (1)        FN             |          TP

"""
sys.stdout.write("\n-----------------------------------------------------------------\n")


# 성능 평가 지표 (리포트) 확인
rst = classification_report(testy, pred_y)
print(rst)
sys.stdout.write("\n-----------------------------------------------------------------\n")


# 정밀도, 재현율, ROC 곡선, f1-Score
rs = recall_score(testy, pred_y)
f1 = f1_score(testy, pred_y)
ps = precision_score(testy, pred_y)
roc_auc = roc_auc_score(testy, pred_y)

print(f"정밀도 = {ps}\n"
      f"재현율 = {rs}\n"
      f"f1-score = {f1}\n"
      f"roc_auc 점수 = {roc_auc}")

"""
정밀도: 모델이 Positive로 예측한 것 중 실제로 Positive인 비율.    TP / (FP+TP)    -> 열방향 예측 Positive
재현율: 실제 Positive인 것 중 모델이 Positive로 예측한 비율.      TP / (FN+TP)    -> 행방향 실제 Positive
F1-Score
 - 불균형 분류 문제에서 평가 척도로 활용.
 - 데이터가 불균형한 상태에서의 Accuracy는 데이터 편향이 크게 나타나 성능 측정이 어렵기 때문이다.
 - F1-Score는 정밀도와 재현율의 조화평균을 구한 척도.

AUC : Area Under the Curve
 - ROC 곡선 아래의 면적 > 모델이 Positive와 Negative를 얼마나 잘 구분하는지 평가하는 척도
 - 1.0 : 완벽한 분류 성능
 - 0.5 : 무작위 분류 성능
 - < 0.5: 모델이 잘못 예측하고 있는 상태.
 - 0.0 : 완전히 잘못된 분류 성능.
"""




