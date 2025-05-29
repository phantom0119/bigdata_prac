
"""
ColumnTransformer를 활용한 OneHotEncoder 변환 데이터 결합 및 학습 적용 실습
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# 데이터 샘플
sample = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 20, 32],
    'salary': [30000, 45000, 55000, 60000, 70000, 80000, 28000, 45000],
    'color': ['red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue'],
    'target': [0, 1, 1, 0, 1, 1, 0, 1]
})

X = sample.drop(['target'], axis=1)
y = sample.target


encoder = OneHotEncoder(drop='first', sparse_output=False)

process = ColumnTransformer(
    transformers= [
        ('num', StandardScaler(), ['age', 'salary']),
        ('cat', encoder, ['color'])
    ]
)

# process에서 변환된 데이터를 직접 확인하기
check = process.fit_transform(sample)
print(sample)
print(check)



# 모델 적용하는 파이프라인
# 회귀모델로 적용하려면 'classifier'를 'regressor'로 변환
pipline = Pipeline([
    ('preprocessor', process),
    ('classifier', LogisticRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 모델 학습
pipline.fit(X_train, y_train)
pred = pipline.predict(X_test)

print(confusion_matrix(y_test, pred))
print(accuracy_score(y_test, pred))



ret = pd.get_dummies(sample['color'])
print(ret)





