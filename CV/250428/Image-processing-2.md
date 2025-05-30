## 1. MSE

MSE는 평균제곱오차(Mean squared error)입니다.

![image](https://cdn-api.elice.io/api-attachment/attachment/52d2be3004e24e969f6c1869f553784a/MSE.png)

Python과 numpy를 이용해서 MSE를 직접 구현하고 올바른 오차값이 나오는지 확인해 보세요.

정답 데이터와 입력(추론)데이터가 얼마나 차이가 있는지 수치로 표현 할 수 있습니다. 오차함수이기 때문에 값이 작을수록 유사하고, 클수록 다릅니다.

지시사항에 따라 아래와 같은 매개변수와 반환값을 가지는 `MSE()` 함수를 구현합니다.

- 매개변수
  - `y`: 정답 list
  - `y_hat`: 비교 list
- 출력
  - 오차값(소수점 둘째 자리에서 반올림)

## 지시사항

1. 정답 list와 비교 list의 각 요소의 차이를 구합니다.
2. 1에서 구한 값을 제곱하세요.
3. 2에서 구한 값의 평균을 구하세요.
4. 소수점 둘째 자리에서 반올림하세요.
5. 4의 값을 반환하세요.

### Tips!

아래 Python 코드를 참고해 MSE를 직접 구현해보세요.

- 차(뺄셈)

```
80 - 12 # 두 값의 차인 68이 나옵니다.
a = np.array([8, 3, -3])
b = np.array([3, 6, -6])
a - b # 각 요소의 차인 [5, -3,  3]이 나옵니다.

```

- 제곱

```
2 ** 3 # 2의 3제곱인 8이 나옵니다.
5 ** 2 # 5의 2제곱인 25가 나옵니다.

y = np.array([2, 3, 4])
y ** 2 # 각 값의 제곱인 [4, 9, 16]이 나옵니다.

```

- 평균

```
test_list = [1, 2, 3, 4, 5]
np.mean(test_list) # 입력값의 평균인 3이 나옵니다.
```



## 2. CNN 직접 구현하기

**MNIST 데이터셋**
MNIST 데이터셋은 아래와 같이 숫자 0부터 9까지의 수를 손으로 쓴 28 × 28의 이진 이미지 데이터셋입니다.

60,000개의 학습 셋과 10,000개의 테스트셋이 있습니다.

![image](https://cdn-api.elice.io/api-attachment/attachment/88e1a1e6ce7c42478bf7af80badb1e19/image.png)

우리는 이 데이터셋으로 우리만의 CNN을 학습 시켜 숫자 이미지를 분리하는 분류기를 만들고자 합니다.

케라스에는 모델을 쉽게 구현할 수 있는 models, layers와 활성화 함수들이 있는 activations 모듈들이 있습니다.

```
from tensorflow.keras import layers, models, activations

```

**모델 생성 예시**
아래의 코드들을 참고하여 지시상황에 제시된 모델 구조를 갖는 모델을 생성하고 모델의 구조를 출력하세요.

```
# 순차적으로 레이어가 쌓이는 모델 만들기
model = models.Sequential()

# 컨볼루션 레이어 만들기
model.add(layers.Convolution2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))

# 풀링 적용하기
model.add(layers.MaxPooling2D((2, 2)))

# 1차원  텐서로 변환하기
model.add(layers.Flatten())

# FC레이어 만들기
model.add(layers.Dense(64, activation='relu'))

# 모델 구조 출력하기
model.summary()

```

## 지시사항

1. 다음은 간단한 CNN 구조 입니다.
   ![image](https://cdn-api.elice.io/api-attachment/attachment/e165b846a08e410598f8bb5ce9ce94db/image.png)
   keras의 `models`, `layers`, `activations` 모듈을 활용하여 위의 구조를 갖는 CNN을 만들고 모델의 구조를 `summary()`함수를 통해 출력하세요.
2. 모든 Convolution 레이어의 activation함수는 relu를 사용합니다.
3. 마지막에 Dense Layer는 첫 번째 레이어는 relu를 두 번째 레이어는 softmax를 쓰도록 합니다.
4. Convolution 레이어는 패딩을 주지 않습니다.
5. Convolution 레이어는 스트라이드를 주지 않습니다.
6. 모든 Convolution 레이어의 커널 사이즈는 (3, 3)으로 합니다.



## 3. CNN 학습시키기

앞선 실습에서 구현하였던 모델을 이제 학습을 시켜보려고 합니다.

**모델 구조**
![image](https://cdn-api.elice.io/api-attachment/attachment/942f99b77ff44547acb715e804bca8aa/image.png)

**Dateset 로드하기**
keras는 저명한 데이터셋들을 다운로드하고 바로 로드할 수 있도록 해주는 datasets 모듈을 지원합니다.

앞 실습에서 소개된 MNIST 데이터셋을 로드하기 위해선 아래와 같은 코드를 쓰면 됩니다.

```
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

```

**데이터 shape 변경**
MNIST의 데이터는 모두 이진 이미지이기 때문에 image의 shape이 28×28 이 됩니다. 컨볼루션을 위해서는 채널수가 필요하므로 다음과 같이 이미지들을 reshaping 해야 합니다.

```
# 학습 셋은 60000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
train_images = train_images.reshape((60000, 28, 28, 1))

# 테스트 셋은 10000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
test_images = test_images.reshape((10000, 28, 28, 1))

```

**데이터 정규화**
서로 다른 범위의 명도값을 갖더라도 분류가 될 수 있게 하기 위해 정규화를 진행합니다.

```
# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

```

**모델의 컴파일**
모델에 optimizer와 손실 함수, 평가 지표를 설정합니다.
optimizer에선 GD, SGD, Adagrad, Adam 등이 있고 손실 함수는 MSE, categorical cross entropy 등이 있습니다.
마직막으로 모델의 평가지표는 accuracy, bianry accuracy 등등이 있습니다.

**모델 컴파일 예시**

```
# 모델을 컴파일 합니다.
model.compile(optimizer=optimizers.Adam(),
              loss=losses.sparse_categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

```

**모델 학습 함수**

```
model.fit(train_images, train_labels, epochs=1)

```

**모델 평가 함수**

```
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

```

**모델 추론 함수**

```
confidence = model.predict(test_img.reshape((1, row, col, channel)), verbose=2)

```

## 지시사항

1. 앞선 실습을 참고하여 모델 구조를 선언하세요.
2. 모델을 학습하고, 평가하세요.
   - 본 실습에는 `Adam`을 optimizer로
     손실 함수는 `sparse categorical crossentropy`를, 평가 지표는 `categorical_accuracy`를 사용하세요.
   - 모델을 학습하고 테스트 데이터 셋에 대한 `loss`값과 `accuracy`를 구해주세요.
   - 학습 epoch는 1로 지정하세요.
3. 학습과 테스트 데이터 수(`train_cnt, test_cnt`)를 조정하여 주어진 이미지 `7.png`를 입력으로 넣었을 때 해당 손글씨 이미지를 올바르게 분류하는지 모델의 예측 결과를 구하세요.



## 4. LeNet 직접 구현하기

LeNet은 CNN을 가장 처음 도입한 적용한 모델입니다.
LeNet의 경우 LeNet-1부터 LeNet-5까지 다양한 버전으로 존재합니다. LeNet-5를 직접 구현하고 이전 실습의 CNN과 어떤 점이 다른지 확인해보세요.

## 지시사항

1. 아래 그림은 LeNet의 구조 입니다.
   ![image](https://cdn-api.elice.io/api-attachment/attachment/6375acd884d74f8ab3cab9dc804d701a/image.png)
2. 아래 표는 LeNet의 각 레이어별 커널, 스트라이드 사이즈와 사용되는 활성화 함수가 적인 표 입니다.
   ![image](https://cdn-api.elice.io/api-attachment/attachment/1d3c8228eb904e6cb151e55a70d264ce/image.png)
3. 위 두 그림을 보고 LeNet을 구현하세요.

### Tips!

- 이전에 만든 CNN 모델과 비슷하지만 다른 부분이 있으니 모델 구조에 유의해서 작성해 주세요.



## 5. LeNet 학습시키기

앞선 실습에서 구현하였던 모델을 이제 학습을 시켜보려고 합니다.

**모델 구조**
![image](https://cdn-api.elice.io/api-attachment/attachment/98b3d5a56c0f414fa887cb0cc69b5ed6/lenet2.jpeg)

이전 실습에서는 CNN을 학습시켰지만 이번엔 LeNet 모델입니다.

같은 데이터셋, 같은 손실 함수, 같은 평가 지표, 같은 epoch, 같은 optimizer를 사용하여 LeNet을 학습시켜봅시다.

두 네트워크의 loss와 accuracy를 비교해보고 어떤것들이 성능의 차이를 가지고 오는지 생각해봅시다.

## 지시사항

1. 앞선 실습을 참고하여 모델 구조를 선언하세요.
2. 모델을 학습하고, 평가하세요.
   - optimizer로`Adam`을
     손실 함수는 `sparse categorical crossentropy`를, 평 가지표는 `categorical_accuracy`를 사용하세요.
   - 모델을 학습하고 테스트 데이터셋에 대한 `loss`값과 `accuracy`를 구해주세요.
   - 학습 epoch는 1로 지정하세요.
3. 학습과 테스트 데이터 수(`train_cnt, test_cnt`)를 조정하여 주어진 이미지 `2.png`를 입력으로 넣었을 때 해당 손글씨 이미지를 올바르게 분류하는지 모델의 예측 결과를 구하세요.



## 6. 셔플링을 진행하여 모델 성능 개선하기

shuffle은 데이터셋을 랜덤으로 섞는 것을 의미합니다.

데이터 순서가 학습에 영항을 끼치지 않도록 데이터 순서를 섞어야 합니다. 매번 epoch 마다 데이터의 순서가 같다면 순서가 학습에 영향을 끼칠 수 있습니다.

0~9 까지의 숫자 데이터가 100개씩 10종류가 있을 때, batch를 10으로 하면 첫 번째 batch에서는 0만 100개, 두 번째 batch에서는 1만 100개 학습하게 됩니다. 이렇게 학습하는 것보다 데이터를 잘 섞어서(shuffle) 학습했을 때 더 높은 성능(낮은 loss)을 얻을 수 있습니다.

**Keras에서 shuffle**
Keras에서 shuffle은 `fit()`함수의 `shuffle=`옵션으로 기능을 활성화 할 수 있습니다.
`shuffle=True`가 되면 Epoch마다 데이터가 랜덤하게 섞이게 되고, `shuffle=False`가 되면 데이터가 섞이지 않습니다.

두 가지 옵션을 줘서 각각의 `fit()`을 해보고 loss를 비교하는 실습을 해봅니다. 코드의 빈 부분을 채우고, 결과를 확인해 보세요.

## 지시사항

1. `results_no_shuffle` 변수에 **shuffle을 하지 않은** 모델을 학습하여 저장하세요.
2. `results_shuffle` 변수에 **shuffle을 한** 모델을 학습하여 저장하세요.