import codecs
import nltk
from nltk.tokenize import word_tokenize


# 실습 환경에 미리 설치가 됨
# nltk.download('punkt')


def count_words(input_text):
    """
    input_text 내 단어들의 개수를 세는 함수
    :param input_text: 텍스트
    :return: dictionary, key: 단어, value: input_text 내 단어 개수
    """
    # <ToDo>: key: 단어, value: input_text 내 단어 개수인 output_dict을 만듭니다.
    output_dict = {}
    word_tokenized_text = word_tokenize(input_text.lower())  # 소문자로 변환하여 단어를 일관되게 처리합니다.
    for word in word_tokenized_text:
        # 단어가 이미 output_dict에 있다면, 해당 단어의 개수를 1 증가시킵니다.
        if word in output_dict:
            output_dict[word] += 1
        # 단어가 output_dict에 없다면, 해당 단어를 추가하고 개수를 1로 설정합니다.
        else:
            output_dict[word] = 1

    return output_dict


def main():
    # 데이터 파일인 'text8_1m_part_aa.txt'을 불러옵니다.
    with codecs.open("/Users/kimdohyeon/건양대학교병원_바이오헬스/Biomedical_AI_Train/MLP/MLP_Practice_01/data/text8_1m_part_aa.txt", "r", "utf-8") as html_f:
        text8_text = "".join(html_f.readlines())
    
    # 데이터 내 단어들의 개수를 세어봅시다.
    word_dict = count_words(text8_text)
    
    # 단어 개수를 기준으로 정렬하여 상위 10개의 단어를 출력합니다.
    top_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    print(top_words)

    return word_dict


if __name__ == "__main__":
    main()
