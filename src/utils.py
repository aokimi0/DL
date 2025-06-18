from typing import List, Tuple
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import numpy as np

def create_exp_dir(exp_name: str, exp_type: str) -> Tuple[str, str, str]:
    fig_dir = os.path.join("fig", exp_type, exp_name)
    out_dir = os.path.join("out", exp_type, exp_name)
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    return "", fig_dir, out_dir


def save_model(model: nn.Module, path: str):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    print(f"模型已保存至: {path}")

def setup_plotting():
    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei',
        'Source Han Sans CN', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False

def plot_performance(loss_history: List[float], accuracy_history: List[float], save_path: str, title: str = "模型训练过程"):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.rcParams['font.sans-serif'] = [
        'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei',
        'Source Han Sans CN', 'DejaVu Sans'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(loss_history, label='验证损失')
    ax1.set_title(title)
    ax1.set_ylabel('平均损失')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(accuracy_history, label='验证准确率', color='orange')
    ax2.set_xlabel('轮次 (Epoch)')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"性能图表已保存至: {save_path}")

def plot_loss(loss_history: List[float], save_path: str, title: str = "模型训练损失"):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    setup_plotting()
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel("迭代次数 (x1000)")
    plt.ylabel("损失")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"损失图表已保存至: {save_path}")

def plot_confusion_matrix(
    cm: torch.Tensor,
    class_names: List[str],
    save_path: str,
    title: str = "混淆矩阵",
    normalize: bool = False,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.numpy()
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm = cm.numpy()
        print('Confusion matrix, without normalization')

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    setup_plotting()
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="真实标签",
        xlabel="预测标签",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
            
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}") 