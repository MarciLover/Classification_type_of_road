import torch

import train
import eval_metric
import settings 

def eval_model(model: torch.nn.Module, 
               test_dataloader: torch.utils.data.DataLoader,
               class_names,
               device: torch.device):
    
    metric = eval_metric.form_metric(class_names, device)
    loss_fn = train.loss_fn

    y_preds = []
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred_logits = model(X)
            y_pred = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            y_preds.append(y_pred.cpu())

            loss += loss_fn(y_pred_logits, y)
            acc += metric(y_pred_logits.argmax(dim=1), y)
        
        y_pred_tensor = torch.cat(y_preds)
        loss /= len(test_dataloader)
        acc /= len(test_dataloader)
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc.item()}, y_pred_tensor
