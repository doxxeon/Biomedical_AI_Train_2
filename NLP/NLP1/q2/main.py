import re

word_counter = dict()
regex = re.compile('[^a-zA-Z]')

# 텍스트 파일을 소문자로 변환 및 숫자 및 특수기호를 제거한 딕셔너리를 만드세요.
with open('MLP/250527/q2/text.txt', 'r') as f: # 실습 1 과 동일한 방식으로 `IMDB dataset`을 불러옵니다.
    for line in f:
        line = line.lower()
        line = regex.sub(' ', line)
        words = line.split()
        for word in words:
            if word:
                if word in word_counter:
                    word_counter[word] += 1
                else:
                    word_counter[word] = 1



# 단어 "the"의 빈도수를 확인해 보세요.
count = word_counter.get('the', 0)

print(count)