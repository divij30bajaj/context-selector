import abc


class DataBuilder(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def build_sentence_level(self):
        return NotImplementedError("Cannot run abstract method")

    @abc.abstractmethod
    def build_document_level(self):
        return NotImplementedError("Cannot run abstract method")