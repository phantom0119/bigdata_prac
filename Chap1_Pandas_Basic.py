# Pandas 기본 활용 연습
import pandas as pd
import numpy as np

df = pd.read_csv('workpython/sale.csv', encoding='euc-kr',
                 header=0, index_col=0 )

#df.info()
#print(df.describe(include='object'))
#print(df.describe())

print(df.tail())