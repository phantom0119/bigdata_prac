

# 대응표본 + 단측검정 테스트 연습
from scipy import stats
import pandas as pd
import numpy as np

# 샘플 데이터 (대응표본은 두 집단의 표본 수가 같아야 한다)
male = np.array([80, 75, 75, 80, 65, 90, 85, 90, 75, 80])       # 가정(남학생 수학점수)
#female = np.array([85 , 65, 65, 75, 80, 70, 65, 65, 80, 90])    # 가정(여학생 수학점수)
female = np.array([90 , 100, 85, 90, 85, 85, 75, 80, 95, 90])

# 신뢰수준과 유의수준
conf = 0.95
alpha = 1-conf

# 대응표본 t검정 수행 (단측검정)
# 파라미터 기준으로  A, B, alternative='less'  --> A가 B보다 작다 = 대립가설
t_stat, p = stats.ttest_rel(male, female, alternative='less')  # <> 'greater'

print(t_stat, p)