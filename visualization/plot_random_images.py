import torch
from typing import List
import matplotlib.pyplot as plt
import random
import settings

def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n_images: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    
    if n_images > 10:
        n_images = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    if seed:
        random.seed(settings.RANDOM_SEED)

    random_samples_idx = random.sample(range(len(dataset)), k=n_images)
    plt.figure(figsize=(20, 8))

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        targ_image_adjust = targ_image.permute(1, 2, 0)

        plt.subplot(1, n_images, i + 1)
        plt.imshow(targ_image_adjust / 255)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
    plt.show()