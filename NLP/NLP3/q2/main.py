import pandas as pd

def cal_partial_freq(texts, emotion):
    partial_freq = dict()
    filtered_texts = texts[texts['emotion']==emotion]
    filtered_texts = filtered_texts['sentence']
    
    # 전체 데이터 내 각 단어별 빈도수를 입력해 주는 부분을 구현하세요.
    for text in filtered_texts:
        words = text.rsrtrip().split()  # 문장을 단어로 분리합니다.
        for word in words:
            if word not in partial_freq:
                partial_freq[word] = 1
            else:
                partial_freq[word] += 1

    # partial_freq 딕셔너리에서 감정별로 문서 내 단어의 빈도수를 계산하여 반환하는 부분을 구현하세요.
    for word in partial_freq:
        partial_freq[word] /= len(filtered_texts)  # 각 단어의 빈도수를 문서의 수로 나누어 확률로 변환

    return partial_freq

def cal_total_freq(partial_freq):
    total = 0
    # partial_freq 딕셔너리에서 감정별로 문서 내 전체 단어의 빈도수를 계산하여 반환하는 부분을 구현하세요.
    
    for freq in partial_freq.values():
        total += freq
    # 전체 단어의 빈도수는 각 단어의 빈도수를 모두 더한 값입니다.
    # 따라서, total 변수에 각 단어의 빈도수를 모두 더한 값을 저장합니다.
    total /= len(partial_freq)  # 전체 단어의 빈도수를 단어의 수로 나누어 평균 확률로 변환
 

    return total

# Emotions dataset for NLP를 불러옵니다.
data = pd.read_csv("/Users/kimdohyeon/건양대학교병원_바이오헬스/Biomedical_AI_Train/MLP/250529/q2/emotions_train.txt", delimiter=';', header=None, names=['sentence','emotion'])

# joy 감정에서 happy 단어의 발생 확률
joy_freq = cal_partial_freq(data, 'joy')
joy_likelihood = joy_freq.get['happy']/cal_total_freq(joy_freq)
print(joy_likelihood)

# sadness 감정에서 happy 단어의 발생 확률
sad_freq = cal_partial_freq(data, 'sadness')
sad_likelihood = sad_freq.get['happy']/cal_total_freq(sad_freq)
print(sad_likelihood)

# surprise 감정에서 can 단어의 발생 확률
sup_freq = cal_partial_freq(data, 'surprise')
sup_likelihood = sup_freq.get['can']/cal_total_freq(sup_freq)
print(sup_likelihood)
