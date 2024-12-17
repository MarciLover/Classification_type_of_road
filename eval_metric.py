from torchmetrics import Accuracy

def form_metric(class_names, device):
    
    metric = Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)
    return metric