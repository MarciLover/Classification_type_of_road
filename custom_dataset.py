import torch
import pandas as pd
from pathlib import Path
import cv2
from torch.utils.data import Dataset
from typing import Tuple, Dict, List

def find_classes(dataframe: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:

    classes = sorted(entry for entry in dataframe.unique())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in dataframe.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class CustomTensorDataset(Dataset):
    # TensorDataset with support of transforms.
    def __init__(self, work_dir: str, rgb_paths: pd.DataFrame, rgb_labels: pd.DataFrame, transform=None) -> None:
        
        rgb_paths.reset_index(drop=True, inplace=True)
        rgb_labels.reset_index(drop=True, inplace=True)

        self.rgb_labels = rgb_labels
        self.paths = Path(work_dir)  / rgb_paths
        self.classes, self.class_to_idx = find_classes(rgb_labels)
        self.transform = transform
        # self.targets = list(rgb_labels)
        self.targets = list(rgb_labels.apply(lambda x: self.class_to_idx[x]))

    def __getitem__(self, idx):
        if idx >= len(self.paths):
            IndexError(f'List index out of range Dataset')
        else:
            img_path = Path(self.paths[idx])
            image = torch.Tensor(cv2.imread(img_path, 1))
            label = self.rgb_labels[idx]

            if self.transform:
                image = self.transform(image.permute(2, 0, 1))

            return image, self.class_to_idx[label]

    def __len__(self):
        return len(self.paths)