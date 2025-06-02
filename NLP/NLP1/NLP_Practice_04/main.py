from gensim.models.fasttext import FastText
import gensim


def compute_similarity(model, word1, word2):
    """
    word1과 word2의 similarity를 구하는 함수
    :param model: fastText model
    :param word1: 첫 번째 단어
    :param word2: 두 번째 단어
    :return: model에 따른 word1과 word2의 cosine similarity
    """
    # <ToDo>: model에 따른 word1과 word2의 cosine similarity를 계산하세요.
    similarity = model.wv.similarity(word1, word2)

    return similarity


def get_word_by_calculation(model, word1, word2, word3):
    """
    단어 벡터들의 연산 결과 추론하는 함수
    연산: word1 - word2 + word3
    :param model: fastText model
    :param word1: 첫 번째 단어로 연산의 시작
    :param word2: 두 번째 단어로 빼고픈 단어
    :param word3: 세 번째 단어로 더하고픈 단어
    :return: 벡터 계산 결과에 가장 알맞는 단어
    """
    # <ToDo>: model을 이용하여 word1 - word2 + word3 결과에 가장 근접한 단어를 찾으세요.
    output_word = model.wv.most_similar(positive=[word1, word3], negative=[word2], topn=1)[0][0]
    # topn=1로 설정하여 가장 유사한 단어 하나만 반환합니다.
    # positive=[word1, word3]는 word1과 word3의 벡터를 더하고, negative=[word2]는 word2의 벡터를 빼는 연산을 수행합니다.
    # 결과적으로 word1 - word2 + word3에 가장 유사한 단어를 찾습니다.
    # most_similar 함수는 (단어, 유사도) 형태의 튜플을 반환하므로, [0][0]을 통해 단어만 추출합니다.
    # 만약 word1, word2, word3 중 하나라도 OOV라면, 해당 단어를 찾을 수 없으므로 예외가 발생할 수 있습니다.
    if output_word is None:
        output_word = "알 수 없음"  # 만약 유사한 단어를 찾을 수 없다면, "알 수 없음"을 반환합니다.
    # 예외 처리를 통해 OOV 단어에 대한 대응을 추가할 수 있습니다.

    return output_word


def get_similar_word_from_oov(model, word1):
    """
    단어가 Out of Vocabulary(OOV)인 경우 그와 유사한 단어를 추론하여 찾는 함수
    만약 단어가 OOV가 아닌 경우 원래 단어를 반환함
    :param model: fastText model
    :param word1: 입력 단어
    :return: 알맞은 단어
    """
    # <ToDo>: model을 이용하여 word1의 단어를 찾으세요. 만약 model의 사전에 없는 단어라면 근접한 단어를 찾으세요.
    output_word = word1
    if word1 not in model.wv:
        # 만약 word1이 OOV라면, 가장 유사한 단어를 찾습니다.
        try:
            output_word = model.wv.most_similar(word1, topn=1)[0][0]
        except KeyError:
            output_word = "알 수 없음"  # 만약 유사한 단어를 찾을 수 없다면, "알 수 없음"을 반환합니다.
    # 만약 word1이 OOV가 아니라면, 원래 단어를 그대로 반환합니다.
    else:
        output_word = word1
    # 예외 처리를 통해 OOV 단어에 대한 대응을 추가할 수 있습니다.
    # 만약 word1이 OOV라면, model.wv.most_similar를 사용하여 가장 유사한 단어를 찾습니다.
    # 만약 word1이 OOV가 아니라면, 원래 단어를 그대로 반환합니다.
    # model.wv.most_similar는 단어가 OOV일 때 KeyError를 발생시킬 수 있으므로, try-except 블록을 사용하여 예외를 처리합니다.
    # 만약 OOV 단어에 대해 유사한 단어를 찾을 수 없다면, "알 수 없음"을 반환합니다.
    # 만약 OOV 단어에 대해 유사한 단어를 찾을 수 있다면, 해당 단어를 반환합니다.
    # 만약 word1이 OOV가 아니라면, 원래 단어를 그대로 반환합니다.

    return output_word


def main():
    # 학습된 fasttext model의 경로를 적습니다.
    model_path = '/Users/kimdohyeon/건양대학교병원_바이오헬스/Biomedical_AI_Train/MLP/MLP_Practice/MLP_Practice_04/fasttext_model'
    
    # 학습된 fasttext model을 불러옵니다.
    model = FastText.load(model_path)
    
    # 두 단어의 유사도를 찾습니다.
    word1 = "이순신"
    word2 = "원균"
    word1_word2_sim = compute_similarity(model, word1, word2)
    print("{}와/과 {} 유사도: {}".format(word1, word2, word1_word2_sim))
    
    # '대통령'에서 '현대'를 뺀 후 '고대'를 더하면 어떤 단어가 나올까요?
    word1 = "대통령"
    word2 = "현대"
    word3 = "고대"
    cal_result = get_word_by_calculation(model, word1, word2, word3)
    print("{} - {} + {}: {}".format(word1, word2, word3, cal_result))
    
    # '컴퓨터'라는 단어를 fasttext가 알고 있는지 확인합니다.
    oov_word = "컴퓨터"
    oov_word_result = get_similar_word_from_oov(model, oov_word)
    if oov_word == oov_word_result:
        print("단어 '{}'는 fastText가 알고 있음".format(oov_word))
    else:
        print("{}와 근접한 단어: {}".format(oov_word, oov_word_result))
    
    # '캄퓨터'라는 단어는 오타이지만 사람은 '컴퓨터'임을 알 수 있습니다. 
    # 이를 기계도 알게 만들려면 fasttext를 어떻게 사용하면 될까요?
    oov_word = "캄퓨터"
    oov_word_result = get_similar_word_from_oov(model, oov_word)
    if oov_word == oov_word_result:
        print("단어 '{}'는 fastText가 알고 있음".format(oov_word))
    else:
        print("{}와 근접한 단어: {}".format(oov_word, oov_word_result))

    return word1_word2_sim, cal_result, oov_word_result


if __name__ == "__main__":
    main()
