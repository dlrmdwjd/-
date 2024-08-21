# 2-1. 훈련 세트와 테스트 세트

### ✏️ 지도 학습과 비지도 학습

**지도 학습(supervised learning)**

지도 학습 알고리즘은 훈련하기 위한 데이터와 정답이 필요

입력과 타깃을 전달하여 모델을 훈련한 다음 새로운 데이터를 예측하는 데 활용한다. (k-최근접 이웃)

- **입력(input)** : 데이터
- **타깃(target)** : 정답
- **훈련 데이터(training data)** : 입력과 타깃을 합쳐 훈련 데이터라고 부른다.

**비지도 학습(unsupervised learning)**

타깃 없이 입력 데이터만 사용

정답을 사용하지 않으므로 무언가를 맞힐 수 없는 대신, 데이터를 잘 파악하거나 변형하는 데 도움을 준다. 따라서 무엇을 예측하는 것이 아니라 입력 데이터에서 어떤 특징을 찾는 데 주로 활용한다. 

**** 강화학습(reinforcement learning)**

타깃이 아니라 알고리즘이 행동한 결과로 얻은 보상을 사용해 학습

### 🏋️‍♀️ 훈련 세트와 테스트 세트

머신러닝 알고리즘의 성능을 제대로 평가하려면 훈련 데이터와 평가에 사용할 데이터가 각각 달라야 한다. 가장 간단한 방법은 평가를 위해 또 다른 데이터를 준비하거나, 이미 준비된 데이터 중에서 일부를 떼어 내어 활용하는 것이다. 

- **훈련 세트(train set)** : 모델을 훈련할 때 사용하는 데이터. 보통 훈련 세트가 클수록 좋습니다.
- **테스트 세트(test set)** : 평가에 사용하는 데이터. 전체 데이터에서 20~30%를 테스트 세트로 사용하는 경우가 많으며, 전체 데이터가 아주 크다면 1%만 덜어내도 충분할 수 있다.
- **샘플(sample)** : 하나의 데이터

- **샘플링 편향(sampling bias)** : 훈련 세트와 테스트 세트에 샘플이 골고루 섞여 있지 않아 샘플링이 한쪽으로 치우쳐진 경우

### 👩‍💻 소스 코드

**넘파이(numpy)**

파이썬의 대표적인 배열(array) 라이브러리

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
                
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14

# k-최근접 이웃 알고리즘 사용
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()

# 훈련 세트로 0부터 34번째 인덱스까지 사용
train_input = fish_data[:35]
train_target = fish_target[:35]

# 테스트 세트로 35번째부터 마지막 인덱스까지 사용
test_input = fish_data[35:]
test_target = fish_target[35:]

# 샘플링 편향으로 인해 정확도가 0.0으로 나옴
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

# 넘파이
import numpy as np

# 넘파이 배열 활용
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
print(input_arr)
print(input_arr.shape)

# 무작위로 훈련 세트 나누기
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
print(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# scatter plot으로 무작위 훈련/테스트 세트 확인
import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련 및 성능 측정
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

kn.predict(test_input)

test_target
```