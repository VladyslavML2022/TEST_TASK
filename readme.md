# TEST_TASK

## DS Engineer Test task
## Guidance

### Prerequisites

Project has been tested on 
OS
 - Ubuntu 20.04
 - MacOs Ventura 13.4.1

```bash
# clone repository
git clone https://github.com/VladyslavML2022/TEST_TASK.git
cd TEST_TASK
```

### 1. Counting Islands. Classical Algorithms

Move to counting_islands directory
```bash
cd counting_islands/
```

Create conda environment
```bash
conda create --name counting_islands python=3.10.6 numpy=1.24.1
conda activate counting_islands
```
Test solution with default settings
```bash
python count_islands.py
```
Test with specific parameters
```bash
python count_islands.py --n 10 --m 10
```
Please, see python count_islands.py --help for information about parameters
Feel free to test with different parameters

After test solution use this commands to remove conda env
```bash
conda deactivate
conda remove --name counting_islands --all
# exit from dir
cd ..
```

### 2. Regression on the tabular data. General Machine Learning

Move to regression_on_the_tabular_data solution directory
```bash
cd regression_on_the_tabular_data/
```
### This workspace contains the following structure:
 - data - folder with data for training and testing
 - model - folder where models are stored
 - config.yml - config file with hyperparameters for selected algorithm
 - requirements.txt - for setting up environment
 - eda_analysis.ipynb - jupyter notebook with provided eda analysis
 - train,py, predict.py - python scripts for training and testing
 - prediction_result.csv - file with prediction result on test data


Create conda environment and install all dependencies
```bash
conda create --name regression python=3.10.6
conda activate regression
# install all neccessary dependencies from requirements.txt
pip install --upgrade pip && pip install -r ./requirements.txt
```
Test train.py script with default settings
```bash
python train.py
```
Test with specific parameters
```bash
python train.py --data ./data/train.csv --cfg ./config.yml --model_name ./model/lightgbm_model.pickle
```
Please, see python train.py --help for information about parameters description
Feel free to play with different parameters

Test predict.py script with default settings
```bash
python predict.py
```
Test predict script with explicitly specified parameters
```
python predict.py --data ./data/hidden_test.csv --model ./model/lightgbm_model.pickle --prediction_path ./prediction_result.csv
```
Please, see python predict.py --help for information about parameters description
Feel free to play with different parameters


### For reproducing eda_analysis.ipynb jupyter notebook, you may need install jupyter with pip in your env
```bash
# assummed, conda regression env is activated
pip install jupyter
```

After all testing you can remove conda env as follows:
```bash
conda deactivate
conda remove --name regression --all
# exit from dir
cd ..
```

### 3. MNIST Classifier. OOP

Move to mnist_classifier directory
```bash
cd mnist_classifier
```

### This workspace contains the following structure:
 - digit_classifier.py - executed python script with digi_classifier implementation
 - requirements.txt - for setting up env
 - data - folder which contains some images for test
 - models - module with digit classifier interface implementation


Create conda environment and install all dependencies
```bash
conda create --name mnist python=3.10.0
conda activate mnist
# install all neccessary dependencies from requirements.txt
pip install --upgrade pip && pip install -r ./requirements.txt
```
Test digit_classifier.py script with default settings
```bash
python digit_classifier.py
```
Test with specific parameters
```bash
# convolution neural network
python digit_classifier.py --algorithm cnn --img ./data/img_1.jpg
# random forest
python digit_classifier.py --algorithm rf --img ./data/img_1.jpg
# random algorithm
python digit_classifier.py --algorithm rand --img ./data/img_1.jpg
```
Please, see python digit_classifier.py --help for information about parameters description
Feel free to play with different parameters


After all testing you can remove conda env as follows:
```bash
conda deactivate
conda remove --name mnist --all
# exit from dir
cd ..
```