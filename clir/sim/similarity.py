from abc import ABC


class Similarity(ABC):
    def __init__(self):
        pass

    def compute(self, x_en, x_multi):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__.lower()
