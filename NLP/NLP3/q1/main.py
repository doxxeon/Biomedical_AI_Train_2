from sklearn.model_selection import train_test_split

# 파일을 읽어오세요.
# 각 줄을 (문장, 감정) 형태의 튜플로 data 리스트에 저장합니다.
# 각 줄에서 문장과 감정은 ;으로 구분되어 있고 모든 줄의 끝에 존재하는 \n은 제거해야 합니다.
data = []
with open('/Users/kimdohyeon/건양대학교병원_바이오헬스/Biomedical_AI_Train/MLP/250529/q1/emotions_train.txt', 'r') as f:
    for line in f:
        line = line.replace('\n', '')
        data.append(line)
    


# 읽어온 파일을 학습 데이터와 평가 데이터로 분할하세요.
train, test = train_test_split(data, test_size=0.2, random_state=7)


# 학습 데이터셋의 문장과 감정을 분리하세요.
Xtrain = []
Ytrain = []

for sent_train in train:
    sent, emotion = sent_train.split(';')
    Xtrain.append(sent)
    Ytrain.append(emotion)

print(Xtrain)
print(set(Ytrain))

# 평가 데이터셋의 문장과 감정을 분리하세요.
Xtest = []
Ytest = []

for sent_test in test:
    sent, emotion = sent_test.split(';')
    Xtest.append(sent)
    Ytest.append(emotion)

print(Xtest)
print(set(Ytest))