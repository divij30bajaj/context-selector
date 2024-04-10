from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
import nltk

lemmatizer = WordNetLemmatizer()
# nltk.download('wordnet')
# nltk.download('stopwords')


def check_word(tokenizer, word, prediction):
    word = word.lower().strip()
    word = tokenizer(word)[0].lemma_

    top_five_preds = [pred.lower().strip() for pred in prediction]
    top_five_preds = [tokenizer(w)[0].lemma_ for w in top_five_preds if w != '']
    return word not in top_five_preds and not check_synonym(word, top_five_preds)
    # and word not in stopwords.words('english') and not word.isnumeric()


def check_synonym(word, top_five_preds):
    for w in top_five_preds:
        if are_synonyms(word, w):
            return True
    return False


# def filter_blanks(potential_blanks):
#     unique_words = set()
#     filtered_lists = []
#
#     for lst in potential_blanks:
#         if not all(word in unique_words for word in lst):
#             filtered_lists.append(lst)
#             unique_words.update(lst)
#
#     i = 0
#     second_filtered = []
#     while i < len(filtered_lists)-1:
#         word = filtered_lists[i]
#         shuffle_set = [word]
#         while i < len(filtered_lists)-1 and word[-1] in filtered_lists[i+1]:
#             shuffle_set.append(filtered_lists[i+1])
#             i = i + 1
#             word = filtered_lists[i]
#         if len(shuffle_set) > 1:
#             word = random.choice(shuffle_set)
#         second_filtered.append(word)
#         i = i + 1
#     if i == len(filtered_lists)-1:
#         second_filtered.append(filtered_lists[i])
#     return second_filtered


def are_synonyms(word1, word2):
    synonyms1 = set()
    synonyms2 = set()

    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    for synset in synsets1:
        synonyms1.update(synonym.name() for synonym in synset.lemmas())

    for synset in synsets2:
        synonyms2.update(synonym.name() for synonym in synset.lemmas())

    common_synonyms = synonyms1.intersection(synonyms2)
    return len(common_synonyms) > 0

# def find_infrequent_nouns(sentences, translation_table, tokenizer):
#     noun_list = []
#     for sentence in sentences:
#         words = sentence.lower() \
#             .translate(translation_table) \
#             .replace("-", " ") \
#             .split()
#         for n in range(3, 0, -1):
#             for i in range(len(words) - n + 1):
#                 masked_words = ' '.join(words[i:i + n])
#                 if tokenizer(masked_words)[0].pos_ == "PROPN":
#                     noun_list.append(masked_words)
#
#     count = Counter(noun_list)
#     return count
