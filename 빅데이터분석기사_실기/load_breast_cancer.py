

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = load_breast_cancer()
x = pd.DataFrame(df.data, columns=df.feature_names)
y = pd.DataFrame(df.target, columns=['target'])

df_set = pd.concat([x, y], axis=1)

t1_set = x[['worst concave points', 'worst perimeter', 'mean concave points', 'worst radius', 'mean perimeter']]

train_x, test_x, trainy, testy = train_test_split(t1_set, y, test_size=0.25, random_state=5538)

model2 = RandomForestClassifier(n_estimators=100, random_state=538)
model2.fit(train_x, trainy)
pred = model2.predict(test_x)


csv_set = pd.DataFrame({
	'pred' : pred,
	'real' : testy
})


print(csv_set)


