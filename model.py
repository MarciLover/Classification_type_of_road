import torchvision.models as models
from torch import nn
import torch
from torchinfo import summary
from timeit import default_timer as timer

import eval_metric
import train
import settings

def create_model(class_names):
    model = models.resnet18(pretrained=True).to(settings.device)
    model.fc = nn.Linear(512, len(class_names))
    # summary(model_1, input_size=[1, 3, 256, 256])
    return model

def freeze_layers(model, num_freeze_layers):

    if num_freeze_layers > 4 or num_freeze_layers < 0:
        # print(f'number of freeze layers should be in range [0, ... 4]')
        raise ValueError(f'number of freeze layers should be in range [0, ... 4]')

    if num_freeze_layers >= 1:
        for layer in model.layer1:
            # layer.requires_grad = False
            layer.requires_grad_(False)
    if num_freeze_layers >= 2:
        for layer in model.layer2:
            # layer.requires_grad = False
            layer.requires_grad_(False)
    if num_freeze_layers >= 3:
        for layer in model.layer3:
            # layer.requires_grad = False
            layer.requires_grad_(False)
    if num_freeze_layers == 4:
        for layer in model.layer4:
            # layer.requires_grad = False
            layer.requires_grad_(False)     

    return model

def create_optimizer(model):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=settings.LEARNING_RATE)
    return optimizer

def train_model(model, train_dataloader, val_dataloader, class_names):
    start_time = timer()

    model_res, y_pred_val_tensor = train.train(model=model, 
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            optimizer=create_optimizer(model),
                            loss_fn=train.loss_fn,
                            accuracy=eval_metric.form_metric(class_names, settings.device), 
                            epochs=settings.NUM_EPOCHS,
                            device=settings.device)

    end_time = timer()
    learn_time = end_time-start_time
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    
    return model_res, learn_time, y_pred_val_tensor