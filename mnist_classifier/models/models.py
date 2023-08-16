import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from models.conv_base import CNN
from models.base import DigitClassificationInterface


class CNNClassifier(DigitClassificationInterface):

    def __init__(self):
        
        self.model = CNN()


    def _transform_input(self, image):
        
        transformed_image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return transformed_image


    def predict(self, image):

        transformed_input = self._transform_input(image)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(transformed_input).detach().numpy()[0].argmax()
        return prediction
    

class RandomForest_Classifier(DigitClassificationInterface):

    def __init__(self):

        # we need to fit model on toy data in order to call predict function
        self.train_toy_input = np.random.random((100,784))
        self.train_toy_output = np.random.randint(0,10, size=(100,))
        
        self.model = RandomForestClassifier().fit(self.train_toy_input, self.train_toy_output)

    def _transform_input(self, image):
        
        transformed_input = image.reshape(image.shape[0]*image.shape[1])

        return transformed_input
        
    def predict(self, image):

        transformed_input = self._transform_input(image)


        return self.model.predict([transformed_input]).squeeze()


class RandomClassifier(DigitClassificationInterface):

    def _transform_input(self, image):
        # compute upper right coordinates for new image
        upper_right_x  = (image.shape[1] - 10) // 2
        upper_right_y  = (image.shape[0] - 10) // 2

        center_crop_of_the_image = image[upper_right_y:upper_right_y+10, upper_right_x:upper_right_x+10]

        return center_crop_of_the_image

    def predict(self, image):
        
        transformed_input = self._transform_input(image)

        return np.random.randint(0,10)