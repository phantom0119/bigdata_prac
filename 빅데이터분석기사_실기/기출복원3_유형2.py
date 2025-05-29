"""
분류에서 클래스별로 ROC, AUC 성능 평가를 수행해야 한다.



"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


dataset = load_iris()
#print(dataset)

x = dataset.data
y = dataset.target
yy = label_binarize(y, classes=[0, 1, 2])       #  분류 클래스가 3개이기 때문에 OvR 세팅
# yy는 [1,0,0]   [0,1,0]   [0,0,1] 로 구분하게 된다.

# ROC Curve에 필요한 fpr, tpr 값, 그리고 Threshhold (임곗값) 수집
fpr = [None] * 3
tpr = [None] * 3
thr = [None] * 3


scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

print(x)
print(scaled_x)

trainx, testx, trainy, testy = train_test_split(scaled_x, y, test_size=0.3, random_state=55)

for i in range(3):
    # 각 클래스별로 0 또는 1의 분류가 가능하므로 이를 학습에 적용 후, ROC, AUC 계산
    model = DecisionTreeClassifier(criterion='gini', random_state=55).fit(scaled_x, yy[:,i])
    fpr[i], tpr[i], thr[i] = roc_curve(yy[:,i], model.predict_proba(scaled_x)[:,1])
    print(auc(fpr[i], tpr[i]))

#
# pred = model.predict(testx)
# proba = model.predict_proba(testx)
# rst_set = pd.DataFrame({
#         'y_pred' : pred,
#         'y_real' : testy
# })
#
# print(rst_set)
#
# conf = confusion_matrix(testy, pred)
# roc_auc = roc_auc_score(testy, pred)
#
# print(roc_auc)