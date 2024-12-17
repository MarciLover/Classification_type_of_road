from torchvision import transforms
from torchvision.transforms import v2

import settings

data_transform_check = transforms.Compose([
    v2.Pad(padding=settings.PAD),
    transforms.RandomRotation((settings.ROTATION_FROM, settings.ROTATION_TILL)),
    # v2.ColorJitter(brightness=.1, 
                   #hue=.1
    #                ),
    transforms.Resize(size=(settings.RESIZE_IMAGE, settings.RESIZE_IMAGE))
])

data_transform_train_val = transforms.Compose([
    v2.Pad(padding=settings.PAD),
    transforms.RandomRotation((settings.ROTATION_FROM, settings.ROTATION_TILL)),
    # v2.ColorJitter(brightness=.1, 
                   #hue=.1
    #                ),
    transforms.Resize(size=(settings.RESIZE_IMAGE, settings.RESIZE_IMAGE)),
    # transforms.ToTensor(), # Если в транфсормацию передается уже тензор - то трансформировать в тензор не нужно.
    transforms.Normalize(mean=settings.MEAN, std=settings.STD)
])

data_transform_test = transforms.Compose([
    v2.Pad(padding=settings.PAD),
    # v2.ColorJitter(brightness=.1, 
                   #hue=.1
    #                ),
    transforms.Resize(size=(settings.RESIZE_IMAGE, settings.RESIZE_IMAGE)),
    transforms.Normalize(mean=settings.MEAN, std=settings.STD)
])

data_transform_visualization = transforms.Compose([
    v2.Pad(padding=settings.PAD),
    # v2.ColorJitter(brightness=.1, 
                   #hue=.1
    #                ),
    transforms.Resize(size=(settings.RESIZE_IMAGE, settings.RESIZE_IMAGE))
])