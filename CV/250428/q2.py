from tensorflow.keras import datasets, layers, models, activations


# 모델 변수를 선언합니다.
model = models.Sequential()

# 모델에 첫 번째 입력 레이어를 추가합니다.
model.add(layers.Convolution2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))


# 아래에 지시상항에 있는 모델 구조가 되도록 나머지 모델 구조를 선언해주세요.
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Convolution2D(64, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Convolution2D(64, (3, 3), activation=activations.relu))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation=activations.relu))
model.add(layers.Dense(10, activation=activations.softmax))


# Model 구조를 출력합니다.
model.summary()
