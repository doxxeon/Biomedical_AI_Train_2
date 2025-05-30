import sys
import numpy
import cv2


def prewitt(img):
    # dst_vertical_edge 변수에 수직 프리윗 필터를 적용하세요.
    dst_vertical_edge = None
    
    # dst_horizontal_edge 변수에 수평 프리윗 필터를 적용하세요.
    dst_horizontal_edge = None
    
    
    # 결과 이미지는 수직/수평 프리윗 필터를 적용시킨 두 이미지의 합을 반환하세요.
    return dst_vertical_edge + dst_horizontal_edge


if __name__ == "__main__":
    # 이미지를 읽어옵니다.
    img = cv2.imread("elice.png", cv2.IMREAD_GRAYSCALE)
    
    # 프리윗 피터를 적용한 결과를 확인해봅니다.
    filtered_img = prewitt(img)
    cv2.imwrite("result.jpg", filtered_img)
