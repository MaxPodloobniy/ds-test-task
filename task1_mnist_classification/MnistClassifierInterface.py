from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, dataset):
        pass

    @abstractmethod
    def predict(self, X):
        pass