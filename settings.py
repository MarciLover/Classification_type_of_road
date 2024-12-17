import os
from pathlib import Path
import json
import torch

path = Path('C:/pythonProject1/Work_files/Classification_task_py/Settings')
config_name = 'config_3.json'

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(path/config_name) as f:
    data = json.load(f)

EXP_NAME = data['EXP_NAME']
RANDOM_SEED = data['RANDOM_SEED']
PATH_RESULTS = data['PATH_RESULTS']
vis_plot = data['vis_plot']

TYPE = data['model']['TYPE']
NUM_FREEZE_LAYERS = data['model']['NUM_FREEZE_LAYERS']

TEST_SIZE = data['data']['TEST_SIZE']
PATH_DATASET_MASK = data['data']['PATH_DATASET_MASK']
PATH_DATASET_INPUT = data['data']['PATH_DATASET_INPUT']
PATH_DATASET_OUTPUT = data['data']['PATH_DATASET_OUTPUT']
DIR_WORK_PATH = data['data']['DIR_WORK_PATH']

DELTA_ACC = data['earlystopping']['DELTA_ACC']
TOLERANCE = data['earlystopping']['TOLERANCE']

BATCH_SIZE_TRAIN = data['train']['BATCH_SIZE']
LEARNING_RATE = data['train']['LEARNING_RATE']
NUM_EPOCHS = data['train']['NUM_EPOCHS']

BATCH_SIZE_VAL = data['val']['BATCH_SIZE']

BATCH_SIZE_TEST = data['test']['BATCH_SIZE']

RESIZE_IMAGE = data['augmentation']['RESIZE_IMAGE']
MEAN = data['augmentation']['MEAN']
STD = data['augmentation']['STD']
PAD = data['augmentation']['PAD']
PAD = data['augmentation']['PAD']
ROTATION_FROM = data['augmentation']['ROTATION_FROM']
ROTATION_TILL = data['augmentation']['ROTATION_TILL']

def make_dir():

    # dir_name = config_name.split('.')[0]
    dir_name = EXP_NAME
    path_res = Path(PATH_RESULTS)
    path_res = path_res / dir_name

    Path.mkdir(path_res, exist_ok=True)

    # if not os.path.isdir(path_res):
    #    os.mkdir(path_res)

    return path_res

path_res = make_dir()