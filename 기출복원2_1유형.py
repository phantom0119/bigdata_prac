
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./workpython/Boston.csv')
newdf = df.sort_values(['crim'], ascending=False, inplace=False)
newdf.reset_index(drop=True, inplace=True)

print(newdf['crim'].iloc[0:10])

rev = newdf.iloc[9].crim
print(rev)
newdf['crim'].iloc[0:10] = rev


q3 = newdf[newdf['age'] >= 80]
print(q3.crim.mean())
