from gensim.models.word2vec import Word2Vec


def compute_similarity(model, word1, word2):
    """
    word1과 word2의 similarity를 구하는 함수
    :param model: word2vec model
    :param word1: 첫 번째 단어
    :param word2: 두 번째 단어
    :return: model에 따른 word1과 word2의 cosine similarity
    """
    # <ToDo>: model에 따른 word1과 word2의 cosine similarity를 계산하세요.
    try:
        similarity = model.wv.similarity(word1, word2)
    except KeyError:
        # 만약 word1 또는 word2가 OOV라면, similarity를 0으로 설정합니다.
        similarity = None
        print(f"단어 '{word1}' 또는 '{word2}'가 OOV입니다. 유사도는 0으로 설정됩니다.")

    return similarity


def get_word_by_calculation(model, word1, word2, word3):
    """
    단어 벡터들의 연산 결과 추론하는 함수
    연산: word1 - word2 + word3
    :param model: word2vec model
    :param word1: 첫 번째 단어로 연산의 시작
    :param word2: 두 번째 단어로 빼고픈 단어
    :param word3: 세 번째 단어로 더하고픈 단어
    :return: 벡터 계산 결과에 가장 알맞는 단어
    """
    # <ToDo>: model을 이용하여 word1 - word2 + word3 결과에 가장 근접한 단어를 찾으세요.
    try:
        output_word = model.wv.most_similar(positive=[word1, word3], negative=[word2], topn=1)[0][0]
    except KeyError:
        # 만약 word1, word2, word3 중 하나라도 OOV라면, 예외를 발생시킵니다.
        output_word = "알 수 없음"
    # topn=1로 설정하여 가장 유사한 단어 하나만 반환합니다.
    # positive=[word1, word3]는 word1과 word3의 벡터를 더하고, negative=[word2]는 word2의 벡터를 빼는 연산을 수행합니다.
    # 결과적으로 word1 - word2 + word3에 가장 유사한 단어를 찾습니다.
    # most_similar 함수는 (단어, 유사도) 형태의 튜플을 반환하므로, [0][0]을 통해 단어만 추출합니다.

    return output_word


def main():
    # 학습된 word2vec model을 불러옵니다.
    model = Word2Vec.load('/Users/kimdohyeon/건양대학교병원_바이오헬스/Biomedical_AI_Train/MLP/MLP_Practice/MLP_Practice_03/data/w2v_model')
    
    # 두 단어의 유사도를 찾습니다.
    word1 = "이순신"
    word2 = "원균"
    word1_word2_sim = compute_similarity(model, word1, word2)
    print("{}와/과 {} 유사도: {}".format(word1, word2, word1_word2_sim))
    
    # '대한민국'에서 '서울'을 뺀 후 '런던'을 더하면 어떤 단어가 나올까요?
    word1 = "대한민국"
    word2 = "서울"
    word3 = "런던"
    cal_result = get_word_by_calculation(model, word1, word2, word3)
    print("{} - {} + {}: {}".format(word1, word2, word3, cal_result))

    return word1_word2_sim, cal_result


if __name__ == "__main__":
    main()
