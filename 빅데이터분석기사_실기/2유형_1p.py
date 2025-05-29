# 빅데이터 분석기사 2유형 연습

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys

"""
1. 이온층의 상태(Class)를 분류하기 위해 데이터세트를 훈련/검증용으로 구분한다.
2. 로지스틱 회귀분석을 이용한 성능분석 결과를 출력하세요.
      --> 다른 분류 모델 테스트도 해보기.
"""

df = pd.read_csv('workpython/Ionosphere.csv', encoding='utf-8', index_col=0)
sys.stdout.write(f"데이터프레임 구성 컬럼 확인\n{df.columns}\n")
sys.stdout.write(f"\n----------------------------------------------------------------------\n")
print(f"데이터프레임 구성 컬럼 정보\n{df.dtypes}\n")
sys.stdout.write(f"\n----------------------------------------------------------------------\n")
sys.stdout.write(f"데이터셋 Row 확인\n{df.head(10)}\n")



