
# 주제. iris 데이터를 이용한 붓꽃의 분류 (이산형 변수 분류)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


iris = load_iris()
irisdata = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['target'])
irisdata['target'] = irisdata['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

input = irisdata[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].to_numpy()

output = irisdata['target'].to_numpy()


# test_size : 전체 데이터 중 검증 데이터의 비율.  (전체 150 Rows라면  150*0.3 = 45)
trainx, testx, trainy, testy = train_test_split(input, output, test_size=0.3, random_state=55)

# 데이터 표준화 모듈  (Z-Score =  (x-u)/s )
scaler = StandardScaler()
scaler.fit(trainx)

trainx_scale = scaler.transform(trainx)
testx_scale = scaler.transform(testx)

print(trainx_scale[:5])
print(testx_scale[:5])


model = DecisionTreeClassifier(random_state=55)
results = model.fit(trainx_scale, trainy)

# 훈련 집합에 대한 정확도 성능
print(model.score(trainx_scale, trainy) * 100)

# 검증 집합에 대한 정확도
print(model.score(testx_scale, testy) * 100)

# 특성 중요도
print(model.feature_importances_)

plt.figure(figsize=(10, 5))
plot_tree(model)
plt.show()


