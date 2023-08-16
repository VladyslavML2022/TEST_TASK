# import packages
import pandas as pd
from lightgbm import LGBMRegressor
import joblib
import argparse
from omegaconf import OmegaConf
import logging

# setup simple logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# define a function which perform train model
def train_model(path_to_data, path_to_config, model_name):
    
    logger.info("Reading data...")
    data = pd.read_csv(path_to_data)
    
    logger.info("Loading hyperparameters config...")
    config = OmegaConf.load(path_to_config)

    # separate feature matrix and target variable
    X = data.drop('target', axis=1)[['6','7']] # leave only two feature which we have got from eda analysis as most important
    y = data['target']

    # create and build regressor 
    logger.info("Fitting LightGBM regressor...")
    model = LGBMRegressor(**config.lgbm_params, random_state=2023, verbose=-1).fit(X, y)

    

    logger.info("Saving the model as a pickle format...")
    joblib.dump(model, model_name)
    logger.info(f"Model has been saved to {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LightGBM model')
    parser.add_argument('--data', type=str, default='./data/train.csv', help='Path to the train data')
    parser.add_argument('--cfg', type=str, default='./config.yml', help='Path to the hyperparameter config file')
    parser.add_argument('--model_name', type=str, default='./model/lightgbm_model.pickle', help='Model name')
    args = parser.parse_args()

    train_model(args.data, args.cfg, args.model_name)
