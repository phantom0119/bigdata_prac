
import pandas as pd
import numpy as np
import seaborn as sns

# Sample 데이터로 seaborn의 tips 데이터셋 사용
dataset = sns.load_dataset('tips')
#print(dataset.head())
#print(dataset.info())
#print(dataset.describe())

# 문제1. total_bill 변수의 제 1사분위수를 구하고 정수값으로 출력.
# total_bill 변수 확인
#print(dataset['total_bill'])
totbill = dataset['total_bill'] # 특정 열벡터 데이터셋 뽑아두기
print(f"total_bill 1사분위수 : {totbill.quantile(0.25)}\n"
      f"total_bill 2사분위수 : {totbill.quantile(0.5)}\n"
      f"total_bill 3사분위수 : {totbill.quantile(0.75)}\n"
      f"total_bill 4사분위수 : {totbill.quantile(1.0)}")

# 정수로 변환해서 최종 출력
#print(int(totbill.quantile(0.25)))


# 문제2. total_bill 값이 20 이상 25 이하인 데이터의 수를 구하기.
#totbill에서 값이 20 이상 25 이하인 데이터 마스킹 작업
cond1 = totbill >= 20
cond2 = totbill <= 25
result = dataset[cond1 & cond2]  # cond1과 cond2를 모두 만족하는 Row만 추출
#print(result)
#print(len(result))      #데이터의 수 = Row 수
#print(result.shape)     #데이터 행렬 정보 확인  (Row, Column)

# 문제3. tip 변수의 IQR 값을 구하기.
# IQR = 3사분위수 - 1사분위수
#print(dataset['tip'])

#데이터 한 번에 분할해서 담는 방법
q3, q1 = dataset['tip'].quantile([0.75, 0.25])
#print(f"q3={q3}\nq1={q1}")

result = dataset['tip'].quantile(0.75) - dataset['tip'].quantile(0.25)
#print(result)



# 문제4. tip 변수의 상위 10개 값의 총합을 구하고, 소수점을 버려 정수로 출력
head10 = dataset.sort_values(by='tip', ascending=False).head(10)    # Row 상위에서 10개 추출.
#print(head10['tip'].sum())          # 총합구하는 방법1
#print(sum(head10['tip'].head(10)))  # 총합구하는 방법2



# 문제5. 전체 데이터에서 성별이 Female인 비율이 얼마인지 소수점 첫째자리까지 출력.
# 성별 Column = sex.  마스킹 방법으로 필터링 수행.
fm_df = dataset['sex'] == 'Female'
#print(dataset['sex'].values)  # 열벡터 데이터 유형 확인.
print(dataset[fm_df])         # 마스킹 결과를 기반으로 Row 필터링.

# 비율 구하기 =  Target / 전체(total)
rslt = len(dataset[fm_df]) / len(dataset)
print(round(rslt,1))



"""
 # 문제6. 1번 행부터 순서대로 10개 뽑은 후 total_bill 열의 
         평균값을 반올림해서 정수로 변환 후 출력.
"""
# 인덱스를 활용한 데이터 분리
subdf = dataset[:10]          # 0-base Index (인덱싱 번호를 입력)
subdf2 = dataset.loc[0:9, :]  # 1-base Index (인덱스 Row를 그대로 입력)
#print(f"#####  0~9번 인덱스 데이터  #####")
#print(subdf)
#print(subdf2)
#print(f"0~9번 total_bill 평균 = {round(subdf['total_bill'].mean(),0)}")



"""
  # 문제7. 첫번째 행부터 순서대로 50%까지 데이터를 뽑아 tip 변수의 중앙값 계산.
"""
half = int(len(dataset) * 0.5)
sub7 = dataset[:half]
result = sub7['tip'].median()
#print(result)


# 결측치 처리 관련 문제  --> mpg 데이터셋 사용
df = sns.load_dataset('mpg')
#print(df.head())
#print(df.describe())
"""
  # 문제8. 결측값이 존재하는 데이터의 수를 구하시오 (Row)
"""
rslt = df.isnull().sum()
#print(rslt)
#print(rslt.sum())


"""
  # 문제9. horsepower Column의 결측값을 horcepower의 평균으로 대치하고,
      horsepower의 중앙값을 정수로 출력.
"""
# DataFrame 복사
df1 = df.copy()

# horsepower의 평균
a = df['horsepower'].mean()
# None/Nan 값의 위치에 a값을 채운다.
df1['horsepower'] = df['horsepower'].fillna(a)
print(df1.isnull().sum())
print(df1.info())



"""
   # 문제10. horsepower Column에 결측치가 있는 행을 제거.
       첫번째 행부터 순서대로 50%를 추출한 후 해당 데이터의 1사분위수 구하기.
"""
df1 = df.copy()
#print(df1.isnull().sum())
# 결측치가 있는 데이터 (Row) 삭제
# 삭제 후 이전 DataFrame은 지우고 새로 변경된 결과를 기억.
df1.dropna(inplace=True)
#print(df1.isnull().sum())





