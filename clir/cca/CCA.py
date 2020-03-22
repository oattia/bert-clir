from abc import ABC


class CCA(ABC):
    def __init__(self, num_components):
        self.num_components = num_components

    def train(self, x_en, x_multi):
        raise NotImplementedError

    def predict(self, x_en, x_multi):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}_{self.num_components}".lower()
