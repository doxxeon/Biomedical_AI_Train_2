import tensorflow
from tensorflow.keras import datasets, layers, models, activations, losses, optimizers, metrics, utils
import model
import loss
import dataset

if __name__ == "__main__":

    train_images, train_labels = dataset.voc_load_data("./VOC2007/images", "./VOC2007/labels")    

    yolo = model.create_yolo()

    with tensorflow.device("/cpu:0"):
        yolo.compile(optimizer=optimizers.Adam(), loss=loss.yolo_loss)
        yolo.fit(train_images, train_labels, epochs=1, verbose=2)
        result = yolo.evaluate(train_images, train_labels)
        print(result)

    # 수정하지 마세요. 채점에 사용되는 코드입니다.
    print(train_images[0, :, :, :].sum())
    print(train_labels[0, :, :, :].sum())
