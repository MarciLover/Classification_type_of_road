import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
import cv2
import splitfolders
import pandas as pd
import numpy as np

import settings

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def random_image_draw(path_list, how):
    
    random.seed(settings.RANDOM_SEED)
    
    random_image_path = random.choice(path_list)
    img = cv2.imread(random_image_path, how)
    img_as_array = np.asarray(img)
    
    plt.imshow(img_as_array)
    plt.axis(False)
    plt.savefig(settings.path_res / f'{random_image_path.stem}_random_images_draw.png')

def form_dataset(show_image):
    
    random.seed(settings.RANDOM_SEED)
    
    dataset_mask_path = Path(settings.PATH_DATASET_MASK)
    mask_path_list = list(dataset_mask_path.glob("*.png"))
    
    print(f'len_mask_path_list = {len(mask_path_list)}')
    
    mask_path_list = list(dataset_mask_path.glob("*.png"))
    
    splitfolders.ratio(settings.PATH_DATASET_INPUT, output="Dataset_output", seed=settings.RANDOM_SEED, ratio=(.6, .2, .2), group_prefix=None, move=False)
    
    dataset = Path(settings.PATH_DATASET_OUTPUT)
    annotated_data = pd.DataFrame((dataset.glob("*/*/*.jpg")))
    annotated_data['file_name'] = annotated_data[0].apply(lambda x: x.stem)
    annotated_data['marked_split'] = annotated_data[0].apply(lambda x: x.parent.parent.stem)
    annotated_data['label'] = annotated_data[0].apply(lambda x: x.parent.stem)
    annotated_data = annotated_data.drop(0, axis=1)
    
    markup = pd.read_csv(Path(settings.DIR_WORK_PATH) / 'markup.csv')
    
    data_full = markup.copy()
    data_full['file_name'] = data_full['basler_front_left'].apply(lambda x: Path(x).stem)
    data_full = data_full.merge(right=annotated_data, how='left', on='file_name')
    
    prepared_data = data_full.copy()
    prepared_data = prepared_data.rename(columns={'recording': 'directory.bag', 'basler_front_left': 'image_path', 'annotation_basler-front-left_segmentation': 'mask_path'})
    prepared_data.drop(columns=['annotation_basler-front-left_detection', 'frame_name', 'detection', 'segmentation', 'split'], axis=1, inplace=True)
    random_index = random.choice(prepared_data.index)
    path_random_mask = Path(settings.DIR_WORK_PATH) / Path(prepared_data['mask_path'][random_index])
    path_random_image = Path(settings.DIR_WORK_PATH) / Path(prepared_data['image_path'][random_index])
    
    plt.imshow(np.array(cv2.imread(path_random_mask, 0)))
    plt.imshow(np.array(cv2.imread(path_random_image, 1)))
    
    if show_image == True:
        plt.show()
    elif show_image == False:
        pass

    selected_prepared_data = prepared_data.loc[prepared_data['label'].notna()]
    print(f'len_selected_prepared_data = {len(selected_prepared_data)}')

    return selected_prepared_data