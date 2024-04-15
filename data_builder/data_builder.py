import abc


class DataBuilder(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def build_raw_documents(self):
        return NotImplementedError("Cannot run abstract method")

    @abc.abstractmethod
    def build_dataset(self):
        return NotImplementedError("Cannot run abstract method")