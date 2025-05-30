import sys
import cv2
import numpy

# 영상을 입력으로 받아 영상의 히스토그램 영상을 리턴하는 함수를 만드세요.
def draw_histogram(img):
    
    return histogram_img
    

if __name__ == "__main__":
    # 이미지를 읽어옵니다.
    img = cv2.imread("./night.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 결과를 확인합니다.
    histogram_img = draw_histogram(img)
    histogram_img = cv2.resize(histogram_img, (256, 256))
    
    cv2.imwrite('result.jpg', histogram_img) 
