import pandas as pd
from statsmodels.api import Logit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, confusion_matrix
import matplotlib.pyplot as plt


data = pd.read_csv('./workpython/eduhealth.csv', encoding='euc-kr')
"""
#   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   학교ID    9686 non-null   object 
 1   학년      9686 non-null   int64  
 2   건강검진일   9686 non-null   object 
 3   키       9686 non-null   float64
 4   몸무게     9682 non-null   float64
 5   성별      9686 non-null   object 
dtypes: float64(2), int64(1), object(3)
memory usage: 454.2+ KB
None
"""

# (키, 몸무게)를 이용, 성별(남=0, 여=1)을 분류하는 모형 구축.
# statsmodels.api에 있는 Logit 사용
# fit(self, start_params=None, method='newton', maxiter=35, full_output=1, disp=1, callback=None, **kwargs)
data = data.dropna()
trainx = data[['키', '몸무게']]
trainy = data['성별'].map({'남':0, '여':1})

print(trainx.isnull().sum())
model = Logit(endog=trainy, exog=trainx)
result = model.fit()
print(result.summary())

print(result.pvalues)
print(result.conf_int(alpha=0.05)[0])


# 훈련:학습을 7:3으로 나누고 평가한 데이터에 대해 혼동행렬, ROC, AUC 출력
trainX, testX, trainY, testY = train_test_split(trainx, trainy, test_size=0.3, random_state=55)
model = Logit(trainY, trainX)
result = model.fit()

print(result.summary())
pred = result.predict(testX)
ypred = (pred >= 0.5).astype(int)

print(ypred)
print(testY)

fpr, tpr, threshold = roc_curve(ypred, testY)
plt.plot(fpr, tpr)
plt.show()








