from PIL import Image
from PIL import ImageFilter

def sharpening(img):
    # 이미지에 샤프닝 필터 적용
    img_sharpen = img.filter(ImageFilter.SHARPEN)
    return img_sharpen
    
def blur(img):
    # 이미지에 블러 필터 적용
    img_blur = img.filter(ImageFilter.BLUR)
    return img_blur
    
def detect_edge(img):
    # 이미지의 경계선 탐지
    img_edge = img.filter(ImageFilter.FIND_EDGES)
    return img_edge
    
def show_image(img, name):
    img.save(name)

def main():
    img = Image.open("/Users/kimdohyeon/건양대학교병원_바이오헬스/Biomedical_AI_Train/CV/250425/Lenna.png")
    
    # TODO: [지시사항 4번] 지시사항에 따라 적절한 이미지 변환을 수행하세요.
    
    # 이미지 샤프닝 한번 적용하기
    img_sharpen_1 = sharpening(img)
    
    # 이미지 샤프닝 5번 적용하기
    img_sharpen_5 = img
    for _ in range(5):
        img_sharpen_5 = sharpening(img_sharpen_5)
 
    # 이미지 블러 한번 적용하기
    img_blur_1 = blur(img)
    
    # 이미지 블러 5번 적용하기
    img_blur_5 = img
    for _ in range(5):
        img_blur_5 = blur(img_blur_5)
    
    # 이미지 경계선 찾기
    img_edge = detect_edge(img)
    
    print("=" * 50, "샤프닝 한번 적용한 이미지", "=" * 50)
    show_image(img_sharpen_1, "sharpen_1.png")
    
    print("=" * 50, "샤프닝 다섯번 적용한 이미지", "=" * 50)
    show_image(img_sharpen_5, "sharpen_5.png")
    
    print("=" * 50, "블러 한번 적용한 이미지", "=" * 50)
    show_image(img_blur_1, "blur_1.png")
    
    print("=" * 50, "블러 다섯번 적용한 이미지", "=" * 50)
    show_image(img_blur_5, "blur_5.png")
    
    print("=" * 50, "경계선 이미지", "=" * 50)
    show_image(img_edge, "edge.png")
    
    return img_sharpen_1, img_sharpen_5, img_blur_1, img_blur_5, img_edge


if __name__ == "__main__":
    main()
