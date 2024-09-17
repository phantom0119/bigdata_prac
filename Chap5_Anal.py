
import pandas as pd
import numpy as np

data = pd.read_csv('./workpython/train_commerce.csv',
                   index_col=0)

print(data.isnull().info())
print()

# 고유 데이터 값 확인.
print(f"Warehouse_block: "
      f"{data['Warehouse_block'].unique()}")


