

import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('./workpython/height.csv')
print(df.head())

jobdf = df.copy()
jobdf['diff'] = jobdf['h_after'] - jobdf['h_before']

print(round(jobdf['diff'].mean(), 2))

t, p = stats.ttest_ind(jobdf['h_before'], jobdf['h_after'], equal_var=False, alternative='less')

print(round(t,2), round(p,4))

if round(p,4) < 0.05:
    print("귀무가설 기각")
else :
    print("귀무가설 채택")




"""
df = pd.read_csv('./workpython/recordmath.csv')

df.info()

mdf = df[df['sex'] == 'Male']
fdf = df[df['sex'] == 'Female']

acam = round(len(mdf[mdf['academy'] == 1])/len(mdf), 2)
acaf = round(len(fdf[fdf['academy'] == 1])/len(fdf), 2)

print(acam)
print(acaf)

acam_len = len(mdf[mdf['academy'] == 1])
acaf_len = len(fdf[fdf['academy'] == 1])
contin_table = [ [acam_len, acaf_len],  [len(mdf)-acam_len, len(fdf)-acaf_len]]

chi2, pvalue, dof, expected = stats.chi2_contingency(contin_table)

print(chi2, pvalue)
"""