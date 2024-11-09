
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


df = pd.read_csv('./workpython/train_commerce.csv')
print(df.head())
print(df.columns)


df = df.drop(['ID'], axis=1)
df['Reached.on.Time_Y.N'].astype('category')

x = df.drop(['Reached.on.Time_Y.N'], axis=1)
y = df['Reached.on.Time_Y.N']


print(x.dtypes)
print(x['Warehouse_block'].unique())
print(x['Mode_of_Shipment'].unique())
print(x['Product_importance'].unique())
print(x['Gender'].unique())

x['Warehouse_block'].astype('category')
x['Mode_of_Shipment'].astype('category')
x['Product_importance'].astype('category')
x['Gender'].astype('category')

encoder = LabelEncoder()

x['Warehouse_block'] = encoder.fit_transform(x['Warehouse_block'])
x['Mode_of_Shipment'] = encoder.fit_transform(x['Mode_of_Shipment'])
x['Product_importance'] = encoder.fit_transform(x['Product_importance'])
x['Gender'] = encoder.fit_transform(x['Gender'])

encoder = StandardScaler()
scaled_x = encoder.fit_transform(x)

print(scaled_x)

trainx, testx, trainy, testy = train_test_split(scaled_x, y, test_size=0.3, random_state=55)


imp_model = LinearRegression()
imp_model.fit(trainx, trainy)

importance = np.abs(imp_model.coef_)
print(importance)
features = np.argsort(importance)[::-1]
print(features)  # [8 9 2 5 4 6 3 1 7 0]

#4개만 가져와서 학습
xx = x[['Discount_offered', 'Weight_in_gms', 'Customer_care_calls', 'Prior_purchases']]



scaled_x = encoder.fit_transform(xx)
trainx, testx, trainy, testy = train_test_split(scaled_x, y, test_size=0.3, random_state=55)

# model = svm.SVC(kernel='rbf', C=20, gamma=0.1, random_state=55)
# model.fit(trainx, trainy)
# pred = model.predict(testx)
#
# df_result = pd.DataFrame( {
#         'y_pred' : pred,
#         'y_real' : testy
# })
#
# print(df_result)
#
# conf_matrix = confusion_matrix(testy, pred)
# accuracy = accuracy_score(testy, pred)
# error_rate = 1-accuracy
#
# print(conf_matrix)
# print(accuracy, error_rate)



model3 = RandomForestClassifier(n_estimators=120)
model3.fit(trainx, trainy)
pred = model3.predict(testx)

df_rst = pd.DataFrame( {
        'y_pred' : pred,
        'y_real' : testy
})

print(df_rst)
print(accuracy_score(testy, pred))









