# 랜덤포레스트 분류 실습
"""
옵션 설명
1. n_estimators (기본값 100)
 - 생성할 결정 트리(Decision Tree) 개수 설정
 - 많을수록 모델 성능은 개선될 수 있지만, 계산 비용이 증가

2. max_depth  (기본값 None)
 - 결정 트리의 최대 깊이를 지정.
 - 깊을수록 모델이 복잡해져 Overfitting 문제 발생 가능.


3. min_samples_split (기본값 2)
 - 내부 노드를 분할하기 위한 최소 샘플 수.
 - 값이 커질수록 트리가 적게 분할되어 모델 과적합 방지 가능


4. min_samples_leaf (기본값 1)
 - Leaf Node에 있어야 할 최소 샘플 수 지정.
 - 값을 증가시키면 트리의 가지가 작아져 모델이 일반화된다.

5. criterion (기본값 'gini')
 - 결정 트리에서 분할할 때 사용할 평가 기준 지정
 - 'gini' = 지니 불순도 (Gini impurity)
 - 'entropy' = 엔트로피를 사용한 정보 이득을 기반으로 분할

 6. n_jobs (기본값 None)
  - 학습/예측에 사용할 CPU 코어 수 지정.
  - -1  적용 시 사용할 수 있는 모든 CPU 활용.


7. random_state (기본값 None)
 - 랜덤 상수 지정


8. max_features (기본값 'sqrt')
 - 각 결정트리가 분할할 때 고려할 최대 Feature 수 지정
 - 'sqrt' : 제곱근 (전체 Feature 수) 사용.
 - 'log2' : 로그의 피처 수
 - None : 모든 Feature를 사용.
 - 정수 : 사용하고자 하는 Feature 수를 명시
 - 부동소수점 : Feature의 비율


9. bootstrap (기본값 True)
 - 중복을 허용한 샘플링을 사용할지 여부.
 - False = 전체 데이터셋을 사용해 학습한다.


10. oob_score (기본값 Fasle)
  - 부트스트랩 샘플링에서 선택되지 않은 샘플 (Out-of-Bag)을 사용해 모델 성능을 평가할지 여부.
  - True = oob 샘플을 사용해 검증 점수 계산


11. max_leaf_nodes (기본값 None)
 - 트리가 가질 수 있는 최대 리프 노드 수 지정.

12. class_weight (기본값 None)
 - Class 가중치를 조정해 불균형 데이터셋을 처리할 때 도움.
 - None = 모든 클래스가 동일한 가중치
 - 'balanced'  = 클래스 비율에 따라 자동으로 가중치 조정.


13. min_impurity_decrease (기본값 0.0)
 - 노드 분할을 위한 최소 불순도 감소량 지정.
 - 지정한 값보다 불순도 감소가 작은 분할은 수행되지 않는다.


14. warm_start (기본값 False)
 - True = 이전에 학습된 트리를 유지한 상태에서 추가로 트리를 학습


15. ccp_alpha (기본값 0.0)
 - 비용 복잡도 가지치기 (Cost Complexity Pruning)
 - 트리 가지치기 알파 값이며 과적합 방지 목적으로 설정.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('workpython/train_commerce.csv', encoding='utf-8', index_col=0)
df.dropna()

df['Product_importance'] = df['Product_importance'].map({'low':0, 'medium':1, 'high':2})
df['Reached.on.Time_Y.N'] = df['Reached.on.Time_Y.N'].astype('category')    # 종속변수의 범주화

x = df[['Customer_care_calls', 'Prior_purchases', 'Product_importance']].to_numpy()
y = df['Reached.on.Time_Y.N'].to_numpy()

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=10)

# GridSearchCV를 활용한 RandomForest 최적화
grid_params = {
    'n_estimators' : [100, 150, 200, 250],
    'max_depth' : [None, 10, 20],
    'min_samples_split' : [2, 3, 4],
    'min_samples_leaf' : [1, 2, 3],
    'criterion' : ['gini', 'entropy'],
    'max_features' : ['sqrt', 'log2', 1, 2, 3, 0.5],
    'bootstrap' : [True, False],
    'class_weight' : [None, 'balanced']
}

model = RandomForestClassifier()

grid_search = GridSearchCV(model, grid_params, cv=8, n_jobs=-1)
grid_search.fit(trainx, trainy)

print(grid_search.best_params_)

bestmodel = grid_search.best_estimator_
print(bestmodel.score(testx, testy))

