from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

def show_plot(img, title=" "):
    plt.title(title)
    plt.imshow(img)
    plt.savefig("tmp.png")
    plt.show()

def load_image(path, name):
    # TODO: [지시사항 1번] 이미지를 불러오는 함수를 완성하세요
    img = Image.open(os.path.join(path, name))
    return img
    
def main():
    data_path = "/Users/kimdohyeon/건양대학교병원_바이오헬스/Biomedical_AI_Train/CV/250425/dataset/val/dogs"   
    
    
    # 이미지를 불러와 plt를 이용하여 출력합니다
    names = os.listdir(data_path)
    img = load_image(data_path, names[0])
    
    # 원본 이미지를 출력
    show_plot(img, "PIL original image")
    
    
    # TODO: [지시사항 2번] 지시사항에 따라 이미지의 크기를 확인하는 코드를 완성하세요.
    # PIL을 통해 이미지 크기 확인
    pil_size = img.size
    print("PIL을 통한 이미지 크기:", pil_size)
    
    # PIL 이미지를 numpy 배열로 변환
    np_img = np.array(img)
    
    # numpy 배열의 shape 확인
    np_shape = np_img.shape
    print("numpy 배열 shape:", np_shape)
    show_plot(np_img, "Numpy array image")
    
    # TODO: [지시사항 3번] PIL과 numpy를 이용하여 이미지를 다루는 코드를 완성하세요.
    # PIL.Image에서 x=10, y=20 의 픽셀값 가져오기
    pil_pix = img.load()[10, 20]
    
    # numpy 배열에서 x=10, y=20 의 픽셀값을 가져오세요
    np_pix = np_img[10,20]
    print("PIL의 픽셀값: {}, numpy의 픽셀값: {}".format(pil_pix, np_pix))
    
    # PIL을 이용하여 이미지의 크기를 (224,224)로 변형하세요.
    resized_img = img.resize((224, 224))
    
    # resize된 이미지 출력
    show_plot(resized_img, "Resized image")
    print("resize 결과 사이즈:", resized_img.size)
    
    return pil_size, np_img, np_shape, np_pix, resized_img

if __name__ == "__main__":
    main()
