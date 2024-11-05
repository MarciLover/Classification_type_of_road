import os
from pathlib import Path
import json

# json file path
path = Path('C:/Users/pavel.pyrkov/AppData/Local/GitHub/Classification_of_road/Settings') 
# json file name
config_name = 'config_1.json'

# path annotations dataset (road masks) 
PATH_DATASET_MASK = 'C:/pythonProject1/Work_files/all_annotations'
# path markup dataset 
PATH_DATASET_INPUT = 'C:/pythonProject1/Work_files/Dataset_input'
# path where should be placed split dataset
PATH_DATASET_OUTPUT = 'C:/pythonProject1/Work_files/Dataset_output'
# path for full dataset (original)
DIR_WORK_PATH = 'C:/pythonProject1/Work_files/Datasets_other/20230404'
# path where to save results tensorboard
DIR_PATH_TENSORBOARD = 'C:/Users/pavel.pyrkov/AppData/Local/GitHub/Classification_of_road/tensorboard'

with open(path/config_name) as f:
    data = json.load(f)

BATCH_SIZE = data['BATCH_SIZE']
RANDOM_SEED = data['RANDOM_SEED']
NUM_EPOCHS = data['NUM_EPOCHS']
TEST_SIZE = data['TEST_SIZE']
RESIZE_IMAGE = data['RESIZE_IMAGE']
LEARNING_RATE = data['LEARNING_RATE']
DELTA_ACC = data['DELTA_ACC']
TOLERANCE = data['TOLERANCE']
NUM_FREEZE_LAYERS = data['NUM_FREEZE_LAYERS']

def make_dir():

    dir_name = config_name.split('.')[0]

    path_res = Path('C:/Users/pavel.pyrkov/AppData/Local/GitHub/Classification_of_road/Results/')
    path_res = path_res / dir_name
    if not os.path.isdir(path_res):
        os.mkdir(path_res)

    return path_res
