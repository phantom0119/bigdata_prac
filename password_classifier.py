

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

df = pd.read_csv("./workpython/strength_password.csv", on_bad_lines='skip')
newdf = df.copy()

newdf.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 669640 entries, 0 to 669639
Data columns (total 2 columns):
 #   Column    Non-Null Count   Dtype 
---  ------    --------------   ----- 
 0   password  669639 non-null  object
 1   strength  669640 non-null  int64 
dtypes: int64(1), object(1)
memory usage: 10.2+ MB
"""

#print(newdf['strength'].unique())   # [1 2 0]
newdf['strength'] = newdf.strength.astype('category')

y =  newdf['strength']
X =  newdf.drop(['strength'], axis=1, inplace=False)

print(X)

model = RandomForestClassifier()
model.fit(X, y)


