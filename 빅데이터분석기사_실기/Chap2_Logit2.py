
import statsmodels.api as sm   # statsmodel약어. 로지스틱 회귀분석 모형(Logit)
from sklearn.datasets import make_classification  # 랜덤 데이터 생성
import matplotlib.pyplot as plt   # 데이터 시각화
import seaborn as sns             # 데이터 시각화
from sklearn.metrics import log_loss    # 성능분석


"""
    make_classification : 분류 문제에 접근하기 위한 가상의 데이터셋 생성.
        n_features  : 특성 개수
        n_redundant : 중복된 특성 없음(0)
        n_informative : 클래스 구분 시 유용한 특성 개수
        n_clusters_per_class : 클래스당 클러스터 개수
        random_state : 난수 발생 seed 값 
"""
x1, y = make_classification(n_features=1, n_redundant=0
                        ,   n_informative=1, n_clusters_per_class=1
                        ,   random_state=4)

#print(x1[:10])
#print(y[:10])

print(' *****  분석 데이터 시각화  *****')
"""
    scatter.
        - c=y  : 종속변수에 대해 색상을 정의 (클래스에 따라 다른 색상을 적용)
        - s=10 : 데이터값의 크기
        - edgecolor : 테두리 색상
        - linewidth : 선 너비
"""
"""
    kdeplot. (커널 밀도 추정. Kernel Density Estimation)
        - label : 범례 표시 라벨
        - fill : True = 곡선 아래의 영역을 채움
        - ec : 테두리 색
        - fc : 영역의 색
"""
plt.scatter(x1, y, c=y, s=100, edgecolor='k', linewidth=2)  # 산점도
sns.kdeplot(x1[y == 0, :], label = 'y=1', fill=True, ec='red', fc='gray' )


plt.legend()
plt.ylim(-0.2, 1.2 )  # y축 범위 지정

x = sm.add_constant(x1)  # 상수항 추가 (독립변수 배열)
print(x)

logit = sm.Logit(y, x)          # 로지스틱 회귀 모형 구축
results = logit.fit(disp=0)     # 모형 적합, disp: display. 최적화 과정에서 문자열 메시지 생략.

print(f" ###### 로지스틱 회귀분석 모형 레포트  ##### ")
print(results.summary())
print(f" @@@@@ 판별함수 값. 첫 10행 출력 @@@@@")
print(results.fittedvalues[:10])

plt.scatter(x1, y, c=y, s=100, edgecolor='k', lw=2, label="input data")     #입력 데이터 산점도
plt.plot(x1, results.fittedvalues*0.1, label="discriminant function") #판별함수 값
plt.legend()
plt.show()

print(f" #####  분석 모형의 성능분석  ##### ")
ypred = results.predict(x)
print(f"로그 손실(Log Loss) 값 : {log_loss(y, ypred, normalize=False)}")  # 로그 손실 값




