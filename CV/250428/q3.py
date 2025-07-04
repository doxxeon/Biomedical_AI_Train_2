import os
import cv2
import numpy
from tensorflow.keras import datasets, layers, models, activations, losses, optimizers, metrics

# mnist 데이터 셋을 로드합니다.
# 각각 학습셋(이미지, 라벨), 테스트 셋(이미지, 라벨)으로 구성이 되어 있습니다.
data_path = os.path.abspath("./mnist.npz")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)

# 학습과 테스트에 사용할 데이터의 수를 조정합니다.
# 참고, 원래 MNIST의 학습 이미지 셋은 60000개, 테스트 셋은 10000개 입니다.
train_cnt, test_cnt = 5, 1
train_images, train_labels = train_images[:train_cnt], train_labels[:train_cnt]
test_images, test_labels = test_images[:test_cnt], test_labels[:test_cnt]

# 학습 셋은 사용할 갯수만큼 28x28 이진 이미지이므로 reshaping을 해줍니다.
train_images = train_images.reshape((train_cnt, 28, 28, 1))

# 테스트 셋 역시 사용할 갯수만큼 28x28 이진 이미지이므로 reshaping을 해줍니다.
test_images = test_images.reshape((test_cnt, 28, 28, 1))

# 픽셀 값을 0~1 사이로 정규화합니다.
train_images, test_images = train_images / 255.0, test_images / 255.0

# 모델을 구조를 선언하세요.

# 모델을 컴파일 합니다.
adam_optimizer = optimizers.Adam()
loss_function = losses.sparse_categorical_crossentropy
metric = metrics.categorical_accuracy
"""
구현부
"""

# 모델을 학습데이터로 학습하세요.
"""
구현부

"""

# 모델을 평가하세요.
test_loss, test_acc = models.evaluate(test_images, test_labels, verbose=2)


# 학습 결과를 출력합니다.
print("test_loss:", test_loss, "test_acc:", test_acc)

# 모델에 테스트 이미지를 넣고 예측값을 확인해봅니다.
test_img = cv2.imread("7.png", cv2.IMREAD_GRAYSCALE)

# 입력 이미지의 픽셀을 0~1 사이로 정규화 합니다.
test_img = test_img / 255.0
row, col, channel = test_img.shape[0], test_img.shape[1], 1
confidence = model.predict(test_img.reshape((1, row, col, channel)))

for i in range(confidence.shape[1]):
    print(f"{i} 일 확률 = {confidence[0][i]}")

print(f"정답은 : {numpy.argmax(confidence, axis=1)}")
