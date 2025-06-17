import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
from typing import List

from .data_loader import get_cifar10_loaders
from . import models
from .. import utils


def get_model_and_optimizer(model_name, optimizer_name, lr, device):
    model = models.get_model(model_name)
    model.to(device)

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

    return model, optimizer


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    log_interval: int,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def validate_epoch(
    model: nn.Module,
    device: torch.device,
    validation_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    loss_vector: List[float],
    accuracy_vector: List[float],
):
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100.0 * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print(
        f"\nValidation set: Average loss: {val_loss:.4f}, "
        f"Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.0f}%)\n"
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR-10 CNN Experiment")
    parser.add_argument("--model_name", type=str, default="baseline", help="模型 (baseline, resnet18)")
    parser.add_argument("--optimizer", type=str, default="sgd", help="优化器 (sgd, adam)")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--epochs", type=int, default=15, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")
    parser.add_argument("--exp_name", type=str, default="cnn_test", help="实验名称，用于保存图表和模型")
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Running Experiment: {args.exp_name}")
    print(f"Using device: {args.device}")

    train_loader, validation_loader = get_cifar10_loaders(batch_size=args.batch_size)

    model, optimizer = get_model_and_optimizer(
        args.model_name, args.optimizer, args.lr, args.device
    )

    if torch.cuda.device_count() > 1 and args.device == "cuda":
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    print("Model Architecture:")
    print(model)
    print("\nOptimizer:")
    print(optimizer)

    lossv, accv = [], []
    for epoch in range(1, args.epochs + 1):
        train_epoch(
            model, args.device, train_loader, optimizer, criterion, epoch, args.log_interval
        )
        validate_epoch(model, args.device, validation_loader, criterion, lossv, accv)

    output_dir = f"out/cnn/{args.exp_name}"
    figure_path = f"fig/cnn/{args.exp_name}/performance_{args.exp_name}.png"
    
    utils.plot_performance(
        lossv, accv, save_path=figure_path, title=f"CNN - {args.exp_name}"
    )

    os.makedirs(output_dir, exist_ok=True)
    model_save_path = f"{output_dir}/model.pth"
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_save_path)
    else:
        torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main() 