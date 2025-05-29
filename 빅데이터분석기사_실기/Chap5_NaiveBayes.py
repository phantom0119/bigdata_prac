# 나이브 베이즈 ( Naive Bayes ) 분류 모델 실습
"""
  주로 '범주형 (카테고리)' 변수로 구성된 데이터셋에서 적용한다.
  - 나이브 가정 : 주어진 class 조건에서 모든 Feature는 서로 독립적이라고 가정한다.
     --> 현실적으로 그러지 않은 경우가 있겠지만, 분류 성능을 높이는 가정이기도 하다.
     --> 한 특징이 다른 특징에 영향을 미치지 않는다는 가정을 하는 것이다.

Gaussian Naive Bayes
 - Feature가 연속적인 값을 가지는 경우.
 - 가우시안 분포를 가정하여 확률 계산.

Multinomial Naive Bayes
 - Text Classification에서 주로 사용.
 - 특정 단어의 발생 횟수를 기반으로 확률 계산.
 - Feature가 '발생 빈도' 등의 '정수'값을 가질 때 효율적.

Bernoulli Naive Bayes
 - Feature가 [0, 1]의 이진값일 때 적용.
 - ex. E-mail에 특정 단어가 있는지로 스팸 분류.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt



df = pd.read_csv('workpython/train_commerce.csv', encoding='utf-8', index_col=0)
df.dropna()

df['Product_importance'] = df['Product_importance'].map({'low':0, 'medium':1, 'high':2})
df['Reached.on.Time_Y.N'] = df['Reached.on.Time_Y.N'].astype('category')    # 종속변수의 범주화

x = df[['Customer_care_calls', 'Prior_purchases', 'Product_importance']].to_numpy()
y = df['Reached.on.Time_Y.N'].to_numpy()

#print(x)
#print(y)
"""
[[4 3 0]
 [4 2 0]
 [2 4 0]
 ...
 [5 5 0]
 [5 6 1]
 [2 5 0]]
 종속 = [1 1 1 ... 0 0 0]
"""

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=1234)

"""
alpha 옵션 : Laplace Smoothing (라플라스 스무딩)
 - 특정 범주의 값이 없는 경우, 분모가 0이 되는 문제를 해결하기 위한 파라미터
 - 모든 Category의 확률에 alpha 값을 더하는 방식으로 처리.
 - alpha 값이 크면 스무딩 효과가 강해져 모든 범주에 고르게 확률을 부여한다  (기본값 1.0).
 - alpha=0  --> 스무딩을 적용하지 않는다.
"""
model = CategoricalNB(alpha=0.5).fit(trainx, trainy)


pred = model.predict(testx)
#print(model.get_params())
# {'alpha': 0.8, 'class_prior': None, 'fit_prior': True, 'force_alpha': True, 'min_categories': None}
"""
class_prior 옵션 (기본값 None)
 - Class 사전 확률을 수동으로 설정
 - ex) class_prior = [0.5, 0.5]
     --> 두 개의 Class가 동일한 사전 확률을 갖는다고 가정한다.
     
fit_prior 옵션 (기본값 True)
 - 사전 확률을 학습할지 여부를 결정.
 - "사전 확률" = Class의 발생 빈도를 나타내는 확률 척도.
 - False로 설정 시 모든 클래스의 사전 확률을 동일하게 취급하여 계산
     

force_alpha 옵션 (기본값 True)
 - alpha=0인 상태에서도 스무딩을 적용할 수 있다 (True일 때)
 - False이면서 alpha=0이면 카테고리가 존재하지 않는 경우 확률이 0으로 계산될 수 있다.
 

min_categories 옵션 (기본값 None)
 - 각 Feature에 대해 최소 카테고리 수를 설정한다.
 - 특정 Feature에 Category 값이 너무 적으면 성능 저하가 발생할 수 있으므로 수를 강제할 목적으로 사용.
 - ex) min_categories=3  --> 각 Feature는 최소 3개의 카테고리를 가져야 한다.

"""

print(model.score(testx, testy))
print("--------------------------------------------------------------------")


print(f"정확도 = {accuracy_score(testy, pred)}")
print(f"재현율 = {recall_score(testy, pred)}")
print(f"정밀도 = {precision_score(testy, pred)}")
print(f"f1-score = {f1_score(testy, pred)}")
print(f"혼동행렬\n{confusion_matrix(testy, pred)}")


# ROC Curve
fpr, tpr, thresholds = roc_curve(testy, pred)

plt.plot(fpr, tpr)
plt.show()

print(f"AUC = {auc(fpr, tpr)}")