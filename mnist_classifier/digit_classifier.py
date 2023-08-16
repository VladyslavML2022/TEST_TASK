from models import CNNClassifier, RandomForest_Classifier, RandomClassifier
import numpy as np
from PIL import Image
import argparse


class DigitClassifier():

    def __init__(self, algorithm):
        
        if algorithm == 'cnn':

            self.model = CNNClassifier()
        
        elif algorithm == 'rf':
            
            self.model = RandomForest_Classifier()
        
        elif algorithm == 'rand':
            
            self.model = RandomClassifier()
        else:
            raise ValueError("No such algorithm")

    
    def predict(self, image):

        return self.model.predict(image)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test DigitClassifier')
    parser.add_argument('--img', type=str, default='./data/img_1.jpg', help='Select test image for inference')
    parser.add_argument('--algorithm', choices=['cnn','rf','rand'], default='cnn', help='Specify an algorithm')
    args = parser.parse_args()

    image = Image.open(args.img)
    test_input = np.array(image) / 255 # convert to numpy array and normalize

    classifier = DigitClassifier(args.algorithm)

    print(f"Model inference using {args.algorithm} algorithm")
    prediction = classifier.predict(test_input)

    print(f"Predicted value: {prediction}")

    
