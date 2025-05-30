## 1. IoU 함수 구현하기

다음 그림의 빨간색 박스와 초록색 박스의 IoU(Intersection of Union)값을 계산해보겠습니다.
![image](https://cdn-api.elice.io/api-attachment/attachment/1312d7481ae849229d181086e2fa22c0/image.png)

IoU를 계산하는 함수 `intersection_over_union()` 를 지시사항에 따라 구현해보고 결과를 확인해봅시다.

## 지시사항

1. 교집합 부분의 top left 좌표와 bottom right 좌표를 계산하세요.
2. 교집합의 넓이를 구하세요.
3. 박스1의 넓이와 박스2의 넓이를 각각 구하세요.
4. 두 박스의 넓이를 더한 뒤 교집합 영역 넓이를 빼 합영역을 구하세요.
5. 교집합의 넓이와 합영역을 이용해 IoU를 계산 후 반환하세요.



## 2. Average Precision 평가 코드 구현

![img](https://cdn-api.elice.io/api-attachment/attachment/637fa436e9b041049ef23c519a0c931a/image.png)

위의 그림은 강의에서 예시로 들었던 수달 디텍션 알고리즘의 검출 결과에서 알고리즘이 수달이라고 검출한 것들만을 골라낸 결과입니다. 결과에서 첫 번째 컬럼은 IoU를, 두 번째 컬럼은 신뢰도를 의미합니다. 이 디텍션 알고리즘의 성능 평가를 위해 수달 클래스에 대한 `average_precision` 함수를 완성하여 Average Precision 값을 구하려고 합니다.

main 함수 안의 `detection_results`는 위 표에 있는 검출 결과를 신뢰도 값을 기준으로 내림차순 정렬한 것입니다.

`average_precision`함수는`detection_results`와 `ground_truth`를 인자로 받아 average precision을 계산하여 리턴하는 함수입니다.

이 함수의 인자 중 `ground_truth`는 실제 관측되었어야 할 전체 정답의 개수를 의미합니다. 위 문제에서는 실제로 16마리의 수달이 있으므로 `average_precision` 함수의 인자인 `ground_truth`값은 16이 됩니다.

이 `average_precision` 완성해 수달 디텍션의 AP값을 구하는 것이 문제의 목표입니다.
AP를 구할 때는 `scikit-learn` 패키지의 `sklearn.metrics.auc` 함수를 활용해 주세요. `sklearn.metrics.auc` 함수는 auc를 구하는 함수로, 인자로 x와 y좌표를 받습니다. 이 함수를 이용하면, 수달 검출 결과로부터 confidence 레벨에 따른 recall과 precision을 구할 수 있고 이것을 auc 함수에 넣으면 PR 커브 아래의 면적을 구할 수 있으니, 최종적으로는 average precision을 구할 수 있게 됩니다.

## 지시사항

이 알고리즘의 Average Precision 값을 계산하는 `average_precision()` 함수를 완성하세요.

1. Precision-Recall curve를 그리기 위한 thresholds 값은 검출 결과의 confidence 값들로 지정합니다.
2. 각 threshold 값별로 검출됐다고 판단할 수 있는 결과의 전체 개수(`detected`)와 그중에서 클래스를 올바르게 예측한 개수(`tp`)를 구하세요.
3. 각 threshold 별로 precision과 recall 값을 구하세요.

- - 여기서 0으로 나누기 문제에 주의하세요.

1. `sklearn.metric.auc` 함수를 이용하여 커브 면적을 구하세요.

## 참고!

matplolib를 이용하여 recall과 precision 곡선을 그려보고 AP값이 적절히 나온 것 같은지 육안으로 확인해 봅시다.

### auc 커브 그려보기

이 문제의 AUC 커브를 그림으로 그리면 다음과 같습니다.
![img](https://cdn-api.elice.io/api-attachment/attachment/7847aefbe8c14dbeb69ce3c07730f088/Figure_1.png)



## 3. NMS 함수 구현하기

주어진 변수 boxes는 박스의 점수와 좌상단 모서리와 우하단 모서리로 표현되는 바운딩 박스들이 모여있는 numpy.array 배열입니다.

```
[confidence, tl_x, tl_y, br_x, br_y]
```

![image](https://cdn-api.elice.io/api-attachment/attachment/96eed08bf06942a18e511729c630f246/image.png)

인자로 `boxes`와 `iou_threshold`값을 입력으로 받아 Non Max Suppresion을 수행하는 함수 `non_max_suppression()`을 작성하고 NMS 후에 중복 제거가 된 박스들의 개수를 출력해 봅시다.

## 지시사항

1. 박스의 confidence 값들을 내림차순으로 정렬하세요.
2. IoU를 계산하여 박스를 제거하세요.
3. 제거된 박스를 반환하세요.



## 4. Pascal VOC 데이터셋 로드하기

PASCAL VOC(Visual Object Classes) 데이터셋은 다음과 같은 구조로 되어있습니다.

```
VOC2007/
├── images
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── 000003.jpg
└── labels
    ├── 000001.xml
    ├── 000002.xml
    └── 000003.xml
Copy
```

위 구조의 데이터셋을 정해진 포맷대로 읽을 수 있도록 `dataset.py` 파일에 있는 데이터 로더를 만드세요.

**Python 코드 예시**

```
# in_image를 (200, 200)으로 resize 해서 out_img에 저장합니다.
out_img = cv2.resize(in_img, (200, 200))

# (5, 10, 15)크기의 0으로 채워진 벡터를 생성합니다.
label_matrix = numpy.zeros([5, 10, 15])
Copy
```

## 지시사항

1. `dataset.py` 내 `voc_load_data()` 함수를 완성하세요. 아래의 image format과 label format에 맞춰 코드를 반환하세요.

**image format**

이미지 크기는 `(448, 448, 3)`으로 하고 값은 0~1 사이의 소수로 합니다.

예를 들어 이미지 크기가 (700 × 700)일 때, 이 이미지는 (100 × 100) 크기의 셀 (7 × 7) 그리드로 나눠집니다.

- 원본 이미지에서 (50, 50) 좌표는 (0, 0) 번째 그리드에서 (0.5, 0.5) 좌표가 됩니다.
- 원본 이미지에서 (130, 220) 좌표는 (1, 2) 번째 그리드에서 (0.3, 0.2) 좌표가 됩니다.

이 점에 유의해서 xml에 있는 좌표를 데이터 포맷에 맞게 변환하세요.

**label format**

label 크기는 `(7, 7, 25)`로 하고 상세한 포맷은 다음을 따릅니다.

- (7 × 7)은 이미지에 대한 그리드 셀 위치입니다.
- (25) 벡터 index별 데이터는 아래와 같습니다.
  - 0~19 : class id,
    - class id가 0이면 (1, 0, 0, 0, …)
    - class id가 2이면 (0, 0, 1, 0, …)
  - 20 : x, 셀 내에서의 x좌표 (0~1)
  - 21 : y, 셀 내에서의 y좌표 (0~1)
  - 22 : w, 셀 내에서의 w크기 (0~1)
  - 23 : h, 셀 내에서의 h크기(0~1)
  - 24 : Pbject가 있는지 여부 (0 or 1)