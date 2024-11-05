import torch

import learn_models
import eval_metric

def eval_model(model: torch.nn.Module, 
               test_dataloader: torch.utils.data.DataLoader,
               class_names,
               device: torch.device):
    
    accuracy = eval_metric.form_metric(class_names, device)
    loss_fn = learn_models.loss_fn

    y_preds_2 = []
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred_2 = torch.softmax(y_pred, dim=1).argmax(dim=1)
            y_preds_2.append(y_pred_2.cpu())

            loss += loss_fn(y_pred, y)
            acc += accuracy(y_pred.argmax(dim=1), y)
        
        y_pred_tensor_2 = torch.cat(y_preds_2)
        loss /= len(test_dataloader)
        acc /= len(test_dataloader)
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc.item()}, y_pred_tensor_2
