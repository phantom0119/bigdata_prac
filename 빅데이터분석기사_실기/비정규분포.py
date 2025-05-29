import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우의 경우
# plt.rcParams['font.family'] = 'NanumGothic'  # NanumGothic이 설치되어 있는 경우
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# 데이터 생성
np.random.seed(42)
n_samples = 1000

# 1. 정규분포 데이터
normal_data = np.random.normal(loc=0, scale=1, size=n_samples)

# 2. 왜도가 있는 데이터 (오른쪽 꼬리)
skewed_data = np.random.exponential(scale=2.0, size=n_samples)

# 3. 이중봉 분포 데이터
bimodal_data = np.concatenate([
    np.random.normal(loc=-2, scale=0.5, size=n_samples//2),
    np.random.normal(loc=2, scale=0.5, size=n_samples//2)
])

# 4. 균등분포 데이터
uniform_data = np.random.uniform(low=-3, high=3, size=n_samples)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('다양한 데이터 분포 비교', fontsize=16)

# 1. 정규분포
sns.histplot(normal_data, kde=True, ax=axes[0,0])
axes[0,0].set_title('정규분포\n(Normal Distribution)')
axes[0,0].text(0.05, 0.95, f'왜도: {stats.skew(normal_data):.2f}\n첨도: {stats.kurtosis(normal_data):.2f}',
               transform=axes[0,0].transAxes)

# 2. 왜도가 있는 분포
sns.histplot(skewed_data, kde=True, ax=axes[0,1])
axes[0,1].set_title('오른쪽 꼬리가 긴 분포\n(Right-skewed Distribution)')
axes[0,1].text(0.05, 0.95, f'왜도: {stats.skew(skewed_data):.2f}\n첨도: {stats.kurtosis(skewed_data):.2f}',
               transform=axes[0,1].transAxes)

# 3. 이중봉 분포
sns.histplot(bimodal_data, kde=True, ax=axes[1,0])
axes[1,0].set_title('이중봉 분포\n(Bimodal Distribution)')
axes[1,0].text(0.05, 0.95, f'왜도: {stats.skew(bimodal_data):.2f}\n첨도: {stats.kurtosis(bimodal_data):.2f}',
               transform=axes[1,0].transAxes)

# 4. 균등분포
sns.histplot(uniform_data, kde=True, ax=axes[1,1])
axes[1,1].set_title('균등분포\n(Uniform Distribution)')
axes[1,1].text(0.05, 0.95, f'왜도: {stats.skew(uniform_data):.2f}\n첨도: {stats.kurtosis(uniform_data):.2f}',
               transform=axes[1,1].transAxes)

plt.tight_layout()
plt.show()