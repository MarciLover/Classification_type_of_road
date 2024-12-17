import pandas as pd
import torch
import matplotlib.pyplot as plt
import settings
from pathlib import Path

def vis_errors(y_val_pred_tensor_2, val_data_2):

    # compare = pd.DataFrame(zip(y_val_pred_tensor_2, val_data_2.targets))
    # compare[0] = compare[0].astype('int64')
    # compare[2] = False
    # compare.loc[compare[0] == compare[1], 2] = True

    # for i in compare.loc[compare[2] == False, 2].index:
    for i in range(len(y_val_pred_tensor_2)):

        if y_val_pred_tensor_2[i] != val_data_2.targets[i]:

            image = val_data_2[i][0]
            # img_aug = val_data_2[i][0]
            # norm_tensor = torch.zeros([3, 256, 256])
            # norm_tensor[0] = (img_aug[0] + abs(img_aug[0].min())) / (abs(img_aug[0].min() ) + abs(img_aug[0].max()))
            # norm_tensor[1] = (img_aug[1] + abs(img_aug[1].min())) / (abs(img_aug[1].min() ) + abs(img_aug[1].max()))
            # norm_tensor[2] = (img_aug[2] + abs(img_aug[2].min())) / (abs(img_aug[2].min() ) + abs(img_aug[2].max()))

            plt.subplots()
            # print(f'true_class = {val_data_2.classes[val_data_2.__getitem__(i)[1]]}, predicted_class = {val_data_2.classes[y_val_pred_tensor_2[i]]}')
            # plt.imshow((norm_tensor.permute(1, 2, 0)))
        
            plt.imshow(((image / image.max()).permute(1, 2, 0)))
            plt.title(f'true_class = {val_data_2.classes[val_data_2[i][1]]}, predicted_class = {val_data_2.classes[y_val_pred_tensor_2[i]]}')
            local_val_dir = Path('mismatch_of_images_on_validation_selection')
            plt.axis("off")

            Path.mkdir((settings.path_res / local_val_dir), exist_ok=True)
            plt.savefig(settings.path_res / local_val_dir / f'№_mismatch_{i}')