import torchvision.models as models
from torch import nn
import torch
from torchinfo import summary
from timeit import default_timer as timer

import eval_metric
import learn_models
import settings

device = "cuda" if torch.cuda.is_available() else "cpu"

def define_model(class_names):
    model = models.resnet18(pretrained=True).to(device)
    model.fc = nn.Linear(512, len(class_names))
    # summary(model_1, input_size=[1, 3, 256, 256])
    return model

def freeze_layers(model, k):
    if k == 0:
        return model

    elif k == 1:
        for layer in model.layer1:
            # layer.requires_grad = False
            layer.requires_grad_(False)
        return model

    elif k == 2:
        for layer in model.layer1:
            # layer.requires_grad = False
            layer.requires_grad_(False)

        for layer in model.layer2:
            #layer.requires_grad = False
            layer.requires_grad_(False)
        return model
    
    elif k == 3:
        for layer in model.layer1:
            #layer.requires_grad = False
            layer.requires_grad_(False)

        for layer in model.layer2:
                #layer.requires_grad = False
                layer.requires_grad_(False)

        for layer in model.layer3:
                # layer.requires_grad = False
                layer.requires_grad_(False)
        return model
    
    elif k == 4:
        for layer in model.layer1:
            # layer.requires_grad = False
            layer.requires_grad_(False)

        for layer in model.layer2:
                # layer.requires_grad = False
                layer.requires_grad_(False)

        for layer in model.layer3:
                # layer.requires_grad = False
                layer.requires_grad_(False)

        for layer in model.layer4:
                # layer.requires_grad = False
                layer.requires_grad_(False)
        return model
    
    if k > 4 or k < 0:
         print(f'number of freeze layers should be in range [0, ... 4]')

def optimizer_form(model):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=settings.LEARNING_RATE)
    return optimizer

def start_learn(model, train_dataloader, val_dataloader, class_names):
    start_time = timer()

    model_res, y_pred_val_tensor = learn_models.train(model=model, 
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            optimizer=optimizer_form(model),
                            loss_fn=learn_models.loss_fn,
                            accuracy=eval_metric.form_metric(class_names, device), 
                            epochs=settings.NUM_EPOCHS)

    end_time = timer()
    learn_time = end_time-start_time
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    
    return model_res, learn_time, y_pred_val_tensor