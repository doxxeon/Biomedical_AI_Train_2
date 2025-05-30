import sys
import numpy
import cv2

# img에서 특정 색상영역을 추출한 결과를 반환하는 함수를 완성하세요.
def inRange(img, thresholds):
    """
    thresholds는 컬러 채널별 살리고 싶은 명도의 범위가 들어있습니다.
    예를 들어 아래와 같이 입력이 들어온 경우,
    thresholds = [50, 255, 10, 100, 200, 255]
    
    구간 경계선을 포함하여,
    R채널에서는 [50, 255]사이의 픽셀은 255로,범위 밖의 명도는 0으로 표시합니다.
    G채널에서는 [10, 200]사이의 픽셀은 255로,범위 밖의 명도는 0으로 표시합니다.
    B채널에서는 [200, 255]사이의 픽셀은 255로,범위 밖의 명도는 0으로 표시합니다.
    """
    
    return img


if __name__ == "__main__":
    # 컬러 채널별 살리고 싶은 명도의 범위가 주어집니다.
    thresholds = list(map(int, sys.argv[1].split(",")))
    
    # 컬러 이미지를 불러옵니다.
    img = cv2.imread("elice.png", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 특정 색상영역을 추출한 결과를 확인해봅니다.
    colorFiltered = inRange(img, thresholds)
    colorFiltered = cv2.cvtColor(colorFiltered, cv2.COLOR_BGR2RGB)
    cv2.imwrite("result.jpg", colorFiltered)
