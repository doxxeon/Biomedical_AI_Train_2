import sys
import numpy
import cv2

def convolution2D(img, kernel):
    dst = img.copy()
    # 주어진 커널을 입력 img에 회선처리한 이미지 dst를 반환하는 함수를 완성하세요.

    return dst


def prewitt(img):
    vertical_kernel = numpy.array(
        [
            [ -1, 0, 1],
            [ -1, 0, 1],
            [-1, 0, 1]
        ]
    )
    
    horizontal_kernel = numpy.array(
        [
            [ -1, -1, -1],
            [ 0, 0, 0],
            [1, 1, 1]
        ]
    )

    # 프리윗 필터에 회선처리를 적용합니다.
    dst_vertical_edge = convolution2D(img, vertical_kernel)
    dst_horizontal_edge = convolution2D(img, horizontal_kernel)
    
    return dst_vertical_edge + dst_horizontal_edge


if __name__ == "__main__":
    # 이미지를 불러옵니다.
    img = cv2.imread("elice.png", cv2.IMREAD_GRAYSCALE)
    
    # 회선처리 함수를 적용한 프리윗 필터를 적용해봅니다.
    filtered_img = prewitt(img)
    cv2.imwrite("result.jpg", filtered_img)    
