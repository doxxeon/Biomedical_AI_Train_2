import pandas as pd
import numpy as np

def cal_partial_freq(texts, emotion):
    filtered_texts = texts[texts['emotion'] == emotion]
    filtered_texts = filtered_texts['sentence']
    partial_freq = dict()

    # 각 단어의 문서 내 빈도수 계산
    for text in filtered_texts:
        words = text.rstrip.split()
        for word in words:
            if word not in partial_freq:
                partial_freq[word] = 1
            else:
                partial_freq[word] += 1
    # 빈도수를 문서 수로 나누어 확률로 변환
    for word in partial_freq:
        partial_freq[word] /= len(filtered_texts)
    return partial_freq

def cal_total_freq(partial_freq):
    total = 0
    for freq in partial_freq.values():
        total += freq
    if len(partial_freq) > 0:
        total /= len(partial_freq)
    return total

def cal_prior_prob(data, emotion):
    filtered_texts = data[data['emotion'] == emotion]
    # 감정의 사전확률 계산 (로그 적용)
    prior_prob = len(filtered_texts) / len(data)
    return np.log(prior_prob)

def predict_emotion(sent, data):
    emotions = ['anger', 'love', 'sadness', 'fear', 'joy', 'surprise']
    predictions = []
    train_txt = pd.read_csv(data, delimiter=';', header=None, names=['sentence', 'emotion'])

    # sent의 각 감정별 로그 확률을 predictions 리스트에 저장
    for emotion in emotions:
        partial_freq = cal_partial_freq(train_txt, emotion)
        total_freq = cal_total_freq(partial_freq)
        prior_log_prob = cal_prior_prob(train_txt, emotion)
        words = sent.split()
        cond_log_prob = 0
        for word in words:
            word_prob = partial_freq.get(word, total_freq)
            cond_log_prob += np.log(word_prob)
        log_prob = cond_log_prob + prior_log_prob
        predictions.append((emotion, log_prob))
    # 가장 확률이 높은 감정 반환
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[0][0]

# 아래 문장의 예측된 감정을 확인하세요.
test_sent = "i really want to go and enjoy this party"
predicted = predict_emotion(test_sent, "/Users/kimdohyeon/건양대학교병원_바이오헬스/Biomedical_AI_Train/MLP/250529/q3/emotions_train.txt")
print(predicted)