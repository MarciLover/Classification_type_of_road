from pathlib import Path
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import settings

def plot_transformed_images(dataset_path, df_images, df_classes, transform, n_samples, vis_plot):
    random.seed(settings.RANDOM_SEED)
    # random_image_paths = random.sample(list(Path(dataset_path) / df_images.apply(lambda x: Path(x))), k=n)
    # random_image_paths = list(Path(dataset_path) / df_images.sample(n, random_state=seed).reset_index(drop=True).apply(lambda x: Path(x)))
    # random_image_class = (df_classes).sample(n, random_state=seed).reset_index(drop=True)
    random_list_index = random.sample(sorted(list(df_images.index)), n_samples)
    random_image_paths = Path(settings.DIR_WORK_PATH) / df_images.loc[random_list_index]
    random_image_class = df_classes.loc[random_list_index]

    for i in random_image_paths.index:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(Image.open(random_image_paths[i]))
        ax[0].set_title(f"Original \nSize: {(Image.open(random_image_paths[i])).size}")
        ax[0].axis("off")
            
        transformed_image = transform(torch.Tensor(cv2.imread(random_image_paths[i], -1)).permute(2, 0, 1))
             
        # ax[1].imshow(transformed_image.permute(1, 2, 0) / (255))
        ax[1].imshow(transformed_image.permute(1, 2, 0))
        ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
        ax[1].axis("off")

        fig.suptitle(f"Class: {random_image_class[i]}", fontsize=16)
    
        plt.savefig(settings.path_res / f'{i}_plot_transformed_images.png')
    
    if vis_plot == True:
        plt.show()
    elif vis_plot == False:
        pass