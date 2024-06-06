import pandas as pd
import numpy as np

dataset = pd.read_csv('./workpython/usedcars.csv'
                      , index_col=0
                      , encoding='euc-kr'
                      , header=0)

print(dataset.columns)
