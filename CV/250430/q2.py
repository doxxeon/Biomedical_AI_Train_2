import numpy
from sklearn.metrics import auc


# average precision을 반환하는 함수를 완성하세요.
def average_precision(detection_results, ground_truth):
    precisions, recalls = [], []
    # detections_results 이미 문제에서 신뢰도 값을 기준으로 정렬되어 있습니다.
    # 그러므로 for 문을 이용해 가장 높은 Threshold 레벨에서 시작하여
    # 레벨을 점진적으로 낮춥니다.
    for i in range(detection_results.shape[0]):
        # 현재 신뢰도 Threshold
        threshold = detection_results[i, 1]
        
        # detected는 numpy.array 타입으로 detection_results 행 중에서도
        # 현재 신뢰도 Threshold를 만족시키는 결과만 행만 들어가야 합니다
        # numpy의 where 함수를 이용하여 특정 신뢰도 이상이 되는 행들만 골라내봅시다.
        detected = numpy.array()
        
        # 변수 TP는 numpy.array 타입으로 detected 행렬에서 TP에 해당하는 행만
        # 골라낸 것입니다. 
        # 힌트 : numpy.where을 활용하면 쉽게 구할 수 있습니다.
        TP = None
        
        # TP의 개수와, presicion을 구하기 위한 분모인 all detection 개수,
        # recall을 구하기 위한 all ground truth 개수를 알 수 있으므로 
        # 현재 Threshold 레벨에서의 precision과 recall을 구합니다.
        precision = None
        recall = None
        
        precisions.append(precision)
        recalls.append(recall)

    # auc함수에 recall과 precision을 넣어 ap값을 구합니다.
    ap = None
    # matplotlib로 커브 곡선을 눈으로 확인해 보려면 아래 코드를 주석해제 하세요.
    # recalls와 precisions에는 각각 신뢰도 Threshold에 따라 변하는 recall과 대응하는 precision 값들이
    # 들어가야 합니다.
    # from matplotlib import pyplot as plt
    # from elice_utils import EliceUtils
    # print(f"Area under curve = {ap}")
    # plt.plot(recalls, precisions, marker='.')
    # plt.savefig('precision-recall.png', dpi=300)
    # elice_utils = EliceUtils()
    # elice_utils.send_image("precision-recall.png")
    return ap


def main():
    detection_results = numpy.array([
        [1, 0.95],
        [1, 0.91],
        [1, 0.85],
        [1, 0.81],
        [1, 0.78],
        [0, 0.68],
        [1, 0.57],
        [1, 0.45],
        [0, 0.43],
        [0, 0.13],
    ])
    
    # 정의한 함수를 호출한 결과를 소수점 다섯째 자리에서 반올림하여 확인합니다.
    ap = average_precision(detection_results, 16)
    print(round(ap, 4))



if __name__ == "__main__":
    main()
