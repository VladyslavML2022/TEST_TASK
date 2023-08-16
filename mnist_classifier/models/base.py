from abc import ABC, abstractmethod


class DigitClassificationInterface(ABC):

    # abstract method for transforming input in appropriate shape depending on the model type
    @abstractmethod
    def _transform_input(self, image):
        pass

    # absract method for inference
    @abstractmethod
    def predict(self, image):
        pass