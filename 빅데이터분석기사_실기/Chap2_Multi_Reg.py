
"""
    다중 회귀분석
        - 독립변수가 2개 이상.
        - 종속 변수를 예측하기 위한 회귀 분석 방법 중 하나.
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./workpython/seatbelts.csv'
                    , encoding='euc-kr'
                    , header=0
                    , index_col=0)
# 가져온 데이터의 베이스 확인 (첫 5Row: head(), 마지막: tail(n) )
print(dataset.head(6))

