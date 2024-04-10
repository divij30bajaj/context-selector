from data.doc_nmt import DocNMT


def context_builder(document):
    """
    :param document: List of sentences in one document
    :return: List of context sentences, one list per sentence.
    """
    result = [[""], [document[0]]]
    for i in range(2, len(document)):
        result.append([document[i-2], document[i-1]])
    return result


if __name__ == '__main__':
    data_builder = DocNMT(context_builder, dataset="IWSLT17")

    data_builder.build_sentence_level()
    data_builder.build_document_level()
