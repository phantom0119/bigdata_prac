
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

data = pd.read_csv('./workpython/mtcars.csv', index_col=0, encoding='euc-kr')

#print(data.columns)
# 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb'

# 1. 연비 높은 순서대로 정렬  (내림차순)
df = data.sort_values(['mpg'], ascending=[False])

# 2. 연비가 높은 상위 10개의 데이터만 사용
dff = df.head(10)

t1 = dff[dff['carb'] == 2]
t2 = dff[dff['carb'] == 1]


# 3. 마력(hp)의 평균
mt1 = t1['hp'].mean()
mt2 = t2['hp'].mean()

print(abs(mt1-mt2))


data2 = pd.read_csv('./workpython/Ionosphere.csv', encoding='euc-kr', index_col=0)
print(data2.info())
print(data2['Class'].unique())  # bad, good

# 분류 측정 항목(독립 변수)으로 V2와 V1을 제외한다.
x = data2.loc[:, (data2.columns != 'V1') & (data2.columns != 'V2') & (data2.columns != 'Class')]  # Class는 분류 종속 변수이므로 제외.

#print(x.head())
"""
       V3       V4       V5       V6  ...      V31      V32      V33      V34
1  0.99539 -0.05889  0.85243  0.02306  ...  0.42267 -0.54487  0.18641 -0.45300
2  1.00000 -0.18829  0.93035 -0.36156  ... -0.16626 -0.06288 -0.13738 -0.02447
3  1.00000 -0.03365  1.00000  0.00485  ...  0.60436 -0.24180  0.56045 -0.38238
4  1.00000 -0.45161  1.00000  1.00000  ...  0.25682  1.00000 -0.32382  1.00000
5  1.00000 -0.02401  0.94140  0.06531  ... -0.05707 -0.59573 -0.04608 -0.65697

 --> 학습 효율을 높이기 위해 0~1 사이의 값으로 표준화하는 작업 필요.
 --> 표준화하면 평균은 0, 분산은 1이된다.
"""

x = x.to_numpy()
y = data2['Class'].map({'good':1, 'bad':0}).to_numpy()   # 데이터 학습 처리를 위한 Numpy 배열 변환

#print(x[:10])
#print(y[:10])

trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.2, random_state=5538)

# 데이터 표준화 수행
stdmodel = StandardScaler()
stdmodel.fit(trainx)                            # trainx 데이터를 바탕으로 스케일러를 "학습만" 시킴.
trainx_scaler = stdmodel.transform(trainx)
testx_scaler = stdmodel.transform(testx)        # 학습한 Scaler를 바탕으로 주어진 데이터를 변환.

print(testx_scaler)

# 랜덤포레스트 분류 모형
model = RandomForestClassifier( n_estimators=88, random_state=55)
model.fit(trainx_scaler, trainy)
pred = model.predict(testx_scaler)

# 분류 모델에 대한 정확도
print(accuracy_score(testy, pred))

# 혼동 행렬
print(confusion_matrix(testy, pred))

# 성능평가 지표
print(classification_report(testy, pred))


fpr, tpr, thresholds = roc_curve(testy, pred)

print(f"ROC Curve 곡선 아래부분 면적: {auc(fpr, tpr)}")

# plt.plot(fpr, tpr)
# plt.show()

#####################################################################
# 모의시험 1차시
df = pd.read_csv('./workpython/airquality.csv', encoding='euc-kr'
                 , index_col=0 )

# 문제1. 결측치(NaN)를 포함하는 모든 행을 제거하고, Ozone 자료에 대한 상위 60%의 분위 값을 출력.
print(df.isnull().sum())
"""
Ozone      37
Solar.R     7
Wind        0
Temp        0
Month       0
Day         0
"""
#df = df[df['Ozone'].notnull()]
dfnew = df.dropna()
print(dfnew.isnull().sum())

print(dfnew['Ozone'].quantile(0.4))


# 문제2.  5월 측정 자료 (Month=5) 이용.
# 5월 오존량 측정 데이터 24개 중 평균(24.125)보다 큰 값으로 측정된 일수를 구하라.
df_may = df[df['Month'] == 5].dropna()
print(df_may.mean())

t = round(df_may['Ozone'].mean(), 3)
dff = df_may[df_may['Ozone'] > t]
print(len(dff))








