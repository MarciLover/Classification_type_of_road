import torch
from torchmetrics import Accuracy
from torch import nn
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

import earlystopping
import settings

loss_fn = nn.CrossEntropyLoss()

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               accuracy: Accuracy, 
               device: torch.device):
    
    model.train()
    train_loss, train_acc = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        # train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        train_acc += accuracy(y_pred_class, y).item()

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              accuracy: Accuracy,
              device: torch.device):

    model.eval() 
    y_preds_val = []
    val_loss, val_acc = 0, 0
    
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(settings.device), y.to(settings.device)
            val_pred_logits = model(X)
            # y_pred_val = torch.softmax(val_pred_logits, dim=1).argmax(dim=1)
            y_pred_val = torch.argmax(torch.softmax(val_pred_logits, dim=1), dim=1)
            # строка ниже, что с cpu()?
            y_preds_val.append(y_pred_val.cpu())
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()
            # val_pred_labels = val_pred_logits.argmax(dim=1) - данная строка изменена на строку ниже
            # val_pred_labels = torch.argmax(torch.softmax(val_pred_logits, dim=1), dim=1)
            # val_acc += ((val_pred_labels == y).sum().item()/len(val_pred_labels))
            # val_acc += accuracy(val_pred_labels, y).item()
            val_acc += accuracy(y_pred_val, y).item()
            

    y_pred_val_tensor = torch.cat(y_preds_val)        
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc, y_pred_val_tensor

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          accuracy: Accuracy,
          loss_fn: nn.CrossEntropyLoss,
          epochs: int, 
          device: torch.device):
    
    results = {"model_name": model.__class__.__name__,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    early_stopping = earlystopping.EarlyStopping(tolerance=settings.TOLERANCE, min_delta=settings.DELTA_ACC)
    prev_val_acc = 0
    delta_prev = 0
    writer = SummaryWriter('C:/pythonProject1/Work_files/Classification_task_py/tensorbord/config_1')

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           accuracy=accuracy,
                                           device = device)
        
        val_loss, val_acc, y_pred_val_tensor = val_step(model=model,
                                                        dataloader=val_dataloader,
                                                        loss_fn=loss_fn,
                                                        accuracy=accuracy,
                                                        device = device)
        writer.add_scalars("Loss_train/val", {'loss_train': train_loss, 'loss_val': val_loss}, epoch)
        writer.add_scalars("Acc_train/val", {'acc_train': train_acc, 'acc_val': val_acc}, epoch)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        delta = abs(val_acc - prev_val_acc)
        early_stopping(delta, delta_prev)
        print(f'counter = {early_stopping.counter}, | delta_acc = {round(delta, 4)}')

        if early_stopping.early_stop:
            print("We are at epoch:", (epoch + 1))
            break
        
        delta_prev = delta
        prev_val_acc = val_acc

    writer.flush()
    writer.close()
    return results, y_pred_val_tensor

def res_model(res_model):
    model_res = dict()
    length = len(res_model['train_loss']) - 1
    for i in res_model.keys():
        if i != 'model_name':
            model_res[i] = res_model[i][length]
        else: 
            model_res[i] = res_model[i]
    return model_res