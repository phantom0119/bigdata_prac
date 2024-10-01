# SVM 모델에 대한 GridSearch Test 실습

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC



df = pd.read_csv('workpython/train_commerce.csv', index_col=0)
sys.stdout.write(f"데이터 프레임 컬럼 목록\n"
                 f"{df.columns}")
sys.stdout.write("\n-----------------------------------------------------------------\n")
sys.stdout.write(f"컬럼별 데이터 타입 확인\n"
                 f"{df.dtypes}")
sys.stdout.write("\n-----------------------------------------------------------------\n")


# 특정 컬럼 3개를 독립변수 Set으로 만들기
sys.stdout.write(f"독립변수 별 데이터 구조 확인\n"
                 f"Customer_care_calls = {df['Customer_care_calls'].unique()}\n"
                 f"Prior_purchases = {df['Prior_purchases'].unique()}\n"
                 f"Product_importance = {df['Product_importance'].unique()}\n")

# Product_importance의 경우 Object 자료형이므로 학습에 유용하도록 정수 값 인코딩 수행
df['Product_importance'] = df['Product_importance'].map({'low':0, 'medium':1, 'high':2})


dx = df[['Customer_care_calls', 'Prior_purchases', 'Product_importance']]   # 독립변수
dy = df['Reached.on.Time_Y.N']                                              # 종속변수



# 학습/정답 데이터로 사용하기 위해 Numpy 배열로 변환
x = dx.to_numpy()
y = dy.to_numpy()
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
#print(y)
"""
[1 1 1 ... 0 0 0]
"""

# model 생성 과정
"""
 GridSearchCV 방법으로 적용
  - 선택할 수 있는 매개변수 Set을 제공하고, 좀 더 효율적인 파라미터를 선택해 모델을 만드는 방식
  - SVM 모델의 GridSearch 테스트
"""

## 데이터 표준화 작업
trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.3, random_state=55)
scaler = StandardScaler()
trainx_scaled = scaler.fit_transform(trainx)
testx_scaled = scaler.fit_transform(testx)


#SVM 모델인 SVC의 Parameter Set 구성하기
param_grid = {
    'kernel': ['rbf', 'linear', 'sigmoid'],
    'C' : [0.01, 0.05, 0.1, 0.3],
    'gamma' : [0.01, 0.05, 0.1]
}

model = SVC()   # SVM 모델 객체 생성 - 최적화 테스트 할 것이므로 파라미터는 넣지 않음

"""
 cv 옵션 : 교차 검증 (Cross-Validation)
   - 설정한 갑을 바탕으로 N-겹 교차 검증 (N개의 Fold로 나눈 후 각 Fold에서 모델 평가 --> 평균 작업 후 최적의 조합 반환)

n_jobs 옵션 : PC의 CPU 코어를 사용해 병렬작업 처리 설정.
   - -1은 가용한 모든 코어를 사용해 작업을 수행.
"""
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(trainx_scaled, trainy)

sys.stdout.write(f"성능이 우수한 파라미터 선택 결과\n"
                 f"{grid_search.best_params_}")
sys.stdout.write("\n-----------------------------------------------------------------\n")



# 최적의 모델 가져오기
bestmodel = grid_search.best_estimator_
sys.stdout.write(f"최적의 성능 모델의 정확도\n"
                 f"{bestmodel.score(testx_scaled, testy)}")






