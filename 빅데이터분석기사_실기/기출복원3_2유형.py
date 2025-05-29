# 제 3회 기출 복원 2유형 연습

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

iris = load_iris()
x = iris.data
y = iris.target

#print(x, y)  # y값은 [0, 1, 2]

scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.3, random_state=100)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

pred_set = pd.DataFrame( {
        'predict' : pred,
        'actual' : y_test
})

y_pred_proba = model.predict_proba(X_test)[:, 1]

print(pred_set)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
print(accuracy_score(y_test, pred))


model2 = SVC(kernel='sigmoid', random_state=100)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

print(confusion_matrix(pred, y_test))




