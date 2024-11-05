import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import settings

path_res = settings.make_dir()

def plot_loss_curves(results: Dict[str, List[float]]):
    
    loss = results['train_loss']
    val_loss = results['val_loss']
    accuracy = results['train_acc']
    val_accuracy = results['val_acc']
    epochs = range(len(results['train_loss']))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    plt.savefig(path_res / 'функция потерь и метрика.png')

def confusion_matrix(y_pred, targets, class_names, name: str):
    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(preds=y_pred,
                         target=targets)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(12, 9)
    );
    name = Path(name + '.png')
    plt.savefig(path_res / name)

def plot_matrix_barth(data, list_1 : list, nrows, ncols, width, depth):
    
    x = 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, depth))
    fig.tight_layout(h_pad=10, w_pad=10)
    axes = axes.ravel()

    for i in list_1:
        # fig, ax = plt.subplots()
        ax = data[i].plot(
            kind="barh",
            # color=colors[i],
            label=f'{i}',
            ax = axes[x],
            )
            
        ax.set_title(f'Диаграмма параметра "{i}" ' + "\n", fontsize = 14, color = 'Black');
        ax.set_xlabel(f'Значение', fontsize=15)
        ax.set_ylabel('Модель', fontsize=15)

        ax.tick_params(axis="x", rotation=70, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        x += 1

        ax.set_xticks(np.arange(0, 1.05, step=0.05))
        ax.grid(True, axis='x', color='gray', linestyle='--')

def plot_barth(data, i, width, depth):
    
    fig, axes = plt.subplots(figsize=(width, depth))
    fig.tight_layout(h_pad=10, w_pad=10)

    # fig, ax = plt.subplots()
    ax = data[i].plot(
        kind="barh",
        # color=colors[i],
        # label=f'{i}',
        )
            
    ax.set_title(f'Диаграмма параметра "{i}" ' + "\n", fontsize = 14, color = 'Black');
    ax.set_xlabel(f'Значение', fontsize=15)
    ax.set_ylabel('Модель', fontsize=15)

    ax.tick_params(axis="x", rotation=70, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # ax.set_xticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True, axis='x', color='gray', linestyle='--')