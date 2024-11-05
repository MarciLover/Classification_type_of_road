import os
from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import splitfolders
import pandas as pd
import numpy as np

import settings

path_res = settings.make_dir()

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_transformed_images(dataset_path, df_images, df_classes, transform, n, seed):
    random.seed(seed)
    # random_image_paths = random.sample(list(Path(dataset_path) / df_images.apply(lambda x: Path(x))), k=n)
    random_image_paths = list(Path(dataset_path) / df_images.sample(n, random_state=seed).reset_index(drop=True).apply(lambda x: Path(x)))
    random_image_class = (df_classes).sample(n, random_state=seed).reset_index(drop=True)

    for i in range(len(random_image_paths)):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(Image.open(random_image_paths[i])) 
        ax[0].set_title(f"Original \nSize: {(Image.open(random_image_paths[i])).size}")
        ax[0].axis("off")
            
        transformed_image = transform(torch.Tensor(cv2.imread(random_image_paths[i], -1)).permute(2, 0, 1))
             
        ax[1].imshow(transformed_image.permute(1, 2, 0) / (3 * 255))
        ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
        ax[1].axis("off")

        fig.suptitle(f"Class: {random_image_class[i]}", fontsize=16)
    
        plt.savefig(path_res / f'{i}_plot_transformed_images.png')
    # plt.show()

def random_image_draw(path_list, how: int):
    
    random.seed(settings.RANDOM_SEED)
    
    random_image_path = random.choice(path_list)
    # image_class = random_image_path.parent.stem
    img = cv2.imread(random_image_path, how)
    img_as_array = np.asarray(img)
    
    plt.imshow(img_as_array)
    plt.axis(False)
    plt.savefig(path_res / f'{random_image_path.stem} random_images_draw.png')

def form_dataset():
    
    random.seed(settings.RANDOM_SEED)
    
    dataset_mask_path = Path(settings.PATH_DATASET_MASK)
    mask_path_list = list(dataset_mask_path.glob("*.png"))
    
    print(f'len_mask_path_list = {len(mask_path_list)}')
    
    mask_path_list = list(dataset_mask_path.glob("*.png"))
    
    splitfolders.ratio(settings.PATH_DATASET_INPUT, output=settings.PATH_DATASET_OUTPUT, seed=settings.RANDOM_SEED, ratio=(.6, .2, .2), group_prefix=None, move=False)
    
    dataset = Path(settings.PATH_DATASET_OUTPUT)
    # train_dir = dataset / "train"
    # val_dir = dataset / "val"
    # test_dir = dataset / "test"
    # image_path_list = list(dataset.glob("*/*/*.jpg"))
    marked_data = pd.DataFrame((dataset.glob("*/*/*.jpg")))
    marked_data['file_name'] = marked_data[0].apply(lambda x: x.stem)
    marked_data['marked_split'] = marked_data[0].apply(lambda x: x.parent.parent.stem)
    marked_data['label'] = marked_data[0].apply(lambda x: x.parent.stem)
    marked_data = marked_data.drop(0, axis=1)
    
    markup = pd.read_csv(Path(settings.DIR_WORK_PATH) / 'markup.csv')
    
    data_full = markup.copy()
    data_full['file_name'] = data_full['basler_front_left'].apply(lambda x: Path(x).stem)
    data_full = data_full.merge(right=marked_data, how='left', on='file_name')
    
    prepared_data = data_full.copy()
    prepared_data = prepared_data.rename(columns={'recording': 'directory.bag', 'basler_front_left': 'image_path', 'annotation_basler-front-left_segmentation': 'mask_path'})
    prepared_data.drop(columns=['annotation_basler-front-left_detection', 'frame_name', 'detection', 'segmentation', 'split'], axis=1, inplace=True)
    random_index = random.choice(prepared_data.index)
    path_random_mask = Path(settings.DIR_WORK_PATH) / Path(prepared_data['mask_path'][random_index])
    path_random_image = Path(settings.DIR_WORK_PATH) / Path(prepared_data['image_path'][random_index])
    
    plt.imshow(np.array(cv2.imread(path_random_mask, 0)))
    plt.imshow(np.array(cv2.imread(path_random_image, 1)))
    
    selected_prepared_data = prepared_data.loc[prepared_data['label'].notna()]
    
    print(f'len_selected_prepared_data = {len(selected_prepared_data)}')

    return selected_prepared_data