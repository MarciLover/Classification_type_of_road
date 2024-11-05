from torchvision import transforms
from torchvision.transforms import v2

import settings

data_transform_0 = transforms.Compose([
    v2.Pad(padding=[0, 360]),
    transforms.RandomRotation((-5, +5)),
    # v2.ColorJitter(brightness=.1, 
                   #hue=.1
    #                ),
    transforms.Resize(size=(settings.RESIZE_IMAGE, settings.RESIZE_IMAGE))
])

data_transform_1 = transforms.Compose([
    v2.Pad(padding=[0, 360]),
    transforms.RandomRotation((-5, +5)),
    # v2.ColorJitter(brightness=.1, 
                   #hue=.1
    #                ),
    transforms.Resize(size=(settings.RESIZE_IMAGE, settings.RESIZE_IMAGE)),
    # transforms.ToTensor(), # Если в транфсормацию передается уже тензор - то трансформировать в тензор не нужно.
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_transform_2 = transforms.Compose([
    v2.Pad(padding=[0, 360]),
    # v2.ColorJitter(brightness=.1, 
                   #hue=.1
    #                ),
    transforms.Resize(size=(settings.RESIZE_IMAGE, settings.RESIZE_IMAGE)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])