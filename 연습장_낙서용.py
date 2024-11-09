import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 샘플 데이터 생성
data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue', 'green', 'red', 'blue'],
    'target': [0, 1, 1, 0, 1, 1, 0, 1]
})

# 2. 특성과 타겟 분리
X = data[['color']]
y = data['target']

print(data['color'])
print(X)


# 3. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. 원핫인코딩 변환
encoder = OneHotEncoder(sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

print("학습 데이터 원본:", X_train['color'].values)
print("학습 데이터 인코딩:\n", X_train_encoded)

# 5. 모델 학습
model = LogisticRegression()
model.fit(X_train_encoded, y_train)

# 6. 예측 및 평가
predictions = model.predict(X_test_encoded)
accuracy = accuracy_score(y_test, predictions)

print("\n정확도:", accuracy)

# 7. 새로운 데이터로 예측
new_colors = np.array(['red', 'blue', 'green']).reshape(-1, 1)
new_colors_encoded = encoder.transform(new_colors)
new_predictions = model.predict(new_colors_encoded)

print("\n새로운 데이터 예측:")
for color, pred in zip(new_colors.ravel(), new_predictions):
    print(f"색상: {color}, 예측: {pred}")