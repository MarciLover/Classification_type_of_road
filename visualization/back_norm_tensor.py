import torch, settings

def back_norm_tensor(data, index, std = settings.STD, mean = settings.MEAN):
    img_aug = data[index][0]
    norm_tensor = torch.zeros([3, 256, 256])
    for i in range(3):
        # norm_tensor[i] = (img_aug[i] + abs(img_aug[i].min())) / (abs(img_aug[i].min() ) + abs(img_aug[i].max()))
        norm_tensor[i] = (img_aug[i] * std) + mean
    return norm_tensor
