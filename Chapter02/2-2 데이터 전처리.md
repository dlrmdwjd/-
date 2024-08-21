# 2-2. 데이터 전처리

### 🔑 키워드

**스케일(scale)**

두 특성의 값이 놓인 범위가 매우 다른 경우, 두 특성의 스케일이 다르다고 말한다.

데이터를 표현하는 기준(단위)이 다르면 알고리즘이 올바르게 예측할 수 없다. (특히, 알고리즘이 거리 기반일 때)

**데이터 전처리(data preprocessing)**

머신러닝 모델에 훈련 데이터를 주입하기 전에 가공하는 단계. 

샘플 간의 거리에 영향을 많이 받는 알고리즘을 제대로 사용하려면 특성값을 일정한 기준으로 맞춰주는 데이터 전처리가 필요하다. 

**표준점수(standard score)**

z 점수. 훈련 세트의 스케일을 바꾸는 대표적인 방법. 

표준점수를 얻으려면 특성의 평균을 빼고 표준편차로 나눈다. 반드시 훈련 세트의 평균과 표준편차로 테스트 세트를 바꿔야 한다. 

각 특성값이 평균에서 표준편차의 몇 배만큼 떨어져 있는지를 나타낸다. 이를 통해 실제 특성값의 크기와 상관없이 동일한 조건으로 비교할 수 있다.  

**브로드캐스팅**

크기가 다른 넘파이 배열에서 자동으로 사칙 연산을 모든 행이나 열로 확장하여 수행하는 기능

### 👩‍💻 소스 코드

```python

# 데이터 준비
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 넘파이를 활용하여 배열 생성             
import numpy as np

fish_data = np.column_stack((fish_length, fish_weight))
print(fish_data[:5])

fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)

# 사이킷런으로 훈련 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, random_state=42)
    
print(train_input.shape, test_input.shape)
print(train_target.shape, test_target.shape)

# 샘플링 편향이 있는지 확인
print(test_target)

# stratify 매개변수에 타깃 데이터를 전달하여 클래스 비율에 맞게 데이터를 나눔
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)
print(test_target)

# k-최근접 이웃 훈련
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

# 문제의 도미(1) 데이터 결과 확인 -> 빙어(0)로 예측
print(kn.predict([[25, 150]]))

# 산점도 확인
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# k-최근접 이웃은 주변의 샘플 중 다수의 클래스를 예측으로 사용

# 이웃 샘플 확인
distances, indexes = kn.kneighbors([[25, 150]])

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 문제의 도미(1)는 이웃 샘플에 빙어(0)가 더 많음, 즉 가장 가까운 데이터 4개가 빙어(0)

# 이웃 샘플까지의 거리 확인
print(distances)

"""
원인 : x축은 범위가 좁고(10~40), y축은 범위가 넓다(0~1000). 
따라서 y축으로 조금만 멀어져도 거리가 아주 큰 값으로 계산된다. 
이 때문에 오른쪽 위의 도미(1) 샘플이 이웃으로 선택되지 못했다. 
"""

# x축의 범위를 동일하기 0 ~ 1,000 으로 맞추기
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

"""
x축(생선의 길이)은 가장 가까운 이웃을 찾는 데 큰 영향을 미치지 못하고, 
오로지 y축(생선의 무게)만 고려 대상이 되고 있음을 알 수 있다.
"""

# 넘파이를 활용하여 각 특성마다의 평균과 표준편차 계산
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
print(mean, std)

# 원본 데이터에서 평균을 빼고 표준펀차로 나누어 표준점수로 변환 (브로드캐스팅)
train_scaled = (train_input - mean) / std

## 전처리 데이터로 모델 훈련하기 ##

# 표준점수 산점도 확인
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 샘플도 동일한 비율로 변환 (훈련 세트의 mean, std 이용)
new = ([25, 150] - mean) / std

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# k-최근접 이웃 모델로 다시 훈련
kn.fit(train_scaled, train_target)

# 테스트 데이터도 변환
test_scaled = (test_input - mean) / std

# 모델 평가
kn.score(test_scaled, test_target)

# 문제의 도미(1) 샘플로 모델의 예측 출력
print(kn.predict([new]))

# 도미(1)로 예측 성공!

# k-최근접 이웃 다시 구하여 산점도 그리기
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 가까운 샘플이 모두 도미(1)로 변경됨
```