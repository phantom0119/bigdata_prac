

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('./workpython/train_commerce.csv')
print(df.head())
print(df.columns)

x = df[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']]
y = df['Reached.on.Time_Y.N'].astype('category')

encoder = LabelEncoder()

x['Warehouse_block'] = x['Warehouse_block'].astype('category')
x['Mode_of_Shipment'] = x['Mode_of_Shipment'].astype('category')
x['Product_importance'] = x['Product_importance'].astype('category')
x['Gender'] = x['Gender'].astype('category')

x['Warehouse_block'] = encoder.fit_transform(x['Warehouse_block'])
x['Mode_of_Shipment'] =encoder.fit_transform(x['Mode_of_Shipment'])
x['Product_importance'] =encoder.fit_transform(x['Product_importance'])
x['Gender'] =encoder.fit_transform(x['Gender'])

x.info()
print(x.head())
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=100)


model = RandomForestClassifier(n_estimators=100, random_state=100)
model.fit(xtrain, ytrain)

pred = model.predict(xtest)

print(classification_report(pred, ytest))


