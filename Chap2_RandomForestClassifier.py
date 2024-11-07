
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, r2_score

df = load_iris()
x = df.data
y = df.target

# print(x)
# print(y)

scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)

trainx, testx, trainy, testy = train_test_split(scaled_x, y, random_state=42, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(trainx, trainy)
predictions = model.predict(testx)

print(predictions)

conf = classification_report(testy, predictions)
# print(conf)
#
# print(model.feature_importances_)


model2 = LGBMClassifier(n_estimators=100, n_jobs=-1, num_leaves=64)

cross_val = cross_validate(model2, x, y, scoring='accuracy', cv=6,
                           return_train_score=True)

print(f"훈련세트에 대한 정확도 평균 = {cross_val['train_score'].mean()}")
print(f"검증세트에 대한 정확도 평균 = {cross_val['test_score'].mean()}")

# model2.fit(trainx, trainy)
#
# pred = model2.predict(testx)
# conf = classification_report(testy, pred, output_dict=True)


