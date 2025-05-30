import os 
import cv2
import numpy
import matplotlib.pyplot as plt

# Fix seed
import tensorflow as tf
tf.random.set_seed(1)
import numpy as np
np.random.seed(1)

from tensorflow.keras import datasets, layers, models, activations, losses, optimizers, metrics
from tensorflow.keras import utils

# mnist 데이터 셋을 로드합니다.
# 각각 학습셋(이미지, 라벨), 테스트 셋(이미지, 라벨)으로 구성이 되어 있습니다.
data_path = os.path.abspath("./mnist.npz")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)

train_cnt, test_cnt = 5000, 1000
train_images, train_labels = train_images[:train_cnt], train_labels[:train_cnt]
test_images, test_labels = test_images[:test_cnt], test_labels[:test_cnt]

# 학습 셋은 60000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
train_images = train_images.reshape((train_cnt, 28, 28, 1))

# 테스트 셋은 10000개의 28x28 이진 이미지이므로 reshaping을 해줍니다.
test_images = test_images.reshape((test_cnt, 28, 28, 1))

# LeNet의 입력은 32x32 이미지 입니다. 패딩을 주어서 28 x 28에서 32 x 32 이미지로 만듭니다.
train_images = numpy.pad(train_images, [[0, 0], [2,2], [2,2], [0,0]], 'constant')
test_images = numpy.pad(test_images, [[0, 0], [2,2], [2,2], [0,0]], 'constant')
print('train_images :', train_images.shape, type(train_images))
print('test_images :', test_images.shape, type(test_images))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델을 구조를 선언합니다.
model = models.Sequential()
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32, 32, 1)))
model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh'))
model.add(layers.BatchNormalization())
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(120, kernel_size=(1, 1), strides=(5, 5), activation='tanh'))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))


# 모델을 컴파일합니다.
model.compile(loss=losses.sparse_categorical_crossentropy, 
              optimizer=optimizers.Adam(),
              metrics=[metrics.categorical_accuracy])


# 모델을 학습하는 코드를 작성하세요. (shuffle을 하지 않습니다.)
results_no_shuffle = None
no_shuffle_test_loss, no_shuffle_test_acc = model.evaluate(test_images, test_labels)


model.compile(loss=losses.sparse_categorical_crossentropy, 
              optimizer=optimizers.Adam(),
              metrics=[metrics.categorical_accuracy])

# 모델을 학습하는 코드를 작성하세요. (shuffle을 사용해 봅니다.)
results_shuffle = None
shuffle_test_loss, shuffle_test_acc = model.evaluate(test_images, test_labels)

# 코드 작성 후 epoch별 loss를 비교한 결과를 확인해보세요.
print('No Shuffle loss :', [round(x, 3) for x in results_no_shuffle.history['loss']]) 
print('   Shuffle loss :', [round(x, 3) for x in results_shuffle.history['loss']])

print(results_no_shuffle.history['loss'][-1], results_shuffle.history['loss'][-1])


