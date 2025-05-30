import sys
import cv2


# 주어진 영상과 조각 순서를 보고 원본 영상(출력 예시 참고)으로 복원한 영상을 반환하는 함수를 구현하세요.
def solve_puzzle(img, piece_order):
    
    return img
    

if __name__ == "__main__":
    # 퍼즐의 순서가 리스트 형태로 입력됩니다.
    piece_order = list(map(int, list(sys.argv[1].split(','))))
    
    # 이미지를 불러옵니다.
    img = cv2.imread("./puzzle.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 복원 결과를 확인합니다.
    result_img = solve_puzzle(img, piece_order)
    cv2.imwrite('puzzle_result.jpg', result_img) 
