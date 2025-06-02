import re
from sklearn.feature_extraction.text import CountVectorizer

regex = re.compile('[^a-z ]')

with open("/Users/kimdohyeon/건양대학교병원_바이오헬스/Biomedical_AI_Train_2/MLP/MLP5/q1/text.txt", 'r') as f:
    documents = []
    for line in f:
        lowered_sent = line.rstrip()
        filtered_sent = regex.sub('', lowered_sent)
        documents.append(filtered_sent)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

dim = X.shape
print(dim)

words_feature = vectorizer.get_feature_names_out()[:10]
print(words_feature)

idx = vectorizer.vocabulary_("comedy")
print(idx)

vec1 = X[0]
print(vec1)