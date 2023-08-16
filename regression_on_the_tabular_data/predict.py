# import packages for performing inference
import pandas as pd
import joblib
import argparse
import logging

# setup simple logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# define a function which performing model inference on test data
def model_inference(path_to_data, path_to_model, path_to_result):
    
    logger.info("Reading test data...")
    data = pd.read_csv(path_to_data)[['6','7']] # leave only two feature which we have got from eda analysis as most important
     
    logger.info("Loading model...")
    model = joblib.load(path_to_model)

    logger.info("Making inference on test data...")
    prediction = model.predict(data)

    # add prediction column to feature matrix
    logger.info("Saving result...")
    resulted_data = data.assign(prediction=prediction)
    resulted_data.to_csv(path_to_result, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LightGBM model')
    parser.add_argument('--data', type=str, default='./data/hidden_test.csv', help='Path to the test data')
    parser.add_argument('--model', type=str, default='./model/lightgbm_model.pickle', help='Path to the model')
    parser.add_argument('--prediction_path', type=str, default='./prediction_result.csv', help='Path to the resulted file')
    args = parser.parse_args()

    model_inference(args.data, args.model, args.prediction_path)
