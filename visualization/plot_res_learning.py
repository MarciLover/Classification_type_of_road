import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import settings

def plot_loss_curves(results: Dict[str, List[float]]):
    
    train_loss = results['train_loss']
    val_loss = results['val_loss']
    train_accuracy = results['train_acc']
    val_accuracy = results['val_acc']
    epochs = range(len(results['train_loss']))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, val_loss, label='val')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='train')
    plt.plot(epochs, val_accuracy, label='val')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
    plt.savefig(settings.path_res / 'loss_function_and_metric.png')

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
    plt.savefig(settings.path_res / name)

# Для двух функций ниже необходимо создать единый датасет из результатов нескольких моделей. Датасет будет создан в процессе экспериментов. Эта функция возможно будет полезна позже.
def plot_compare_acc(df_models, list_res_train_val_acc : list, nrows, ncols, width, depth):
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, depth))
    fig.tight_layout(h_pad=10, w_pad=10)
    axes = axes.ravel()

    for x, i in enumerate(list_res_train_val_acc):
        # fig, ax = plt.subplots()
        ax = df_models[i].plot(
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

        ax.set_xticks(np.arange(0, 1.05, step=0.05))
        ax.grid(True, axis='x', color='gray', linestyle='--')

def plot_compare_time(df_models, res_time, width, depth):
    
    fig, axes = plt.subplots(figsize=(width, depth))
    fig.tight_layout(h_pad=10, w_pad=10)

    # fig, ax = plt.subplots()
    ax = df_models[res_time].plot(
        kind="barh",
        # color=colors[i],
        # label=f'{i}',
        )
            
    ax.set_title(f'Диаграмма параметра "{res_time}" ' + "\n", fontsize = 14, color = 'Black');
    ax.set_xlabel(f'Значение', fontsize=15)
    ax.set_ylabel('Модель', fontsize=15)

    ax.tick_params(axis="x", rotation=70, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # ax.set_xticks(np.arange(0, 1.05, step=0.05))
    ax.grid(True, axis='x', color='gray', linestyle='--')