import torch

def norm_tensor(data, index):
    img_aug = data.__getitem__(index)[0]
    norm_tensor = torch.zeros([3, 256, 256])
    norm_tensor[0] = (img_aug[0] + abs(img_aug[0].min())) / (abs(img_aug[0].min() ) + abs(img_aug[0].max()))
    norm_tensor[1] = (img_aug[1] + abs(img_aug[1].min())) / (abs(img_aug[1].min() ) + abs(img_aug[1].max()))
    norm_tensor[2] = (img_aug[2] + abs(img_aug[2].min())) / (abs(img_aug[2].min() ) + abs(img_aug[2].max()))
    return norm_tensor
