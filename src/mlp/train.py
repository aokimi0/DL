import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
from typing import List, Tuple

from .data_loader import get_data_loaders
from .model import MLP
from .. import utils


def get_model_and_optimizer(
    model_name: str, optimizer_name: str, lr: float
) -> Tuple[nn.Module, optim.Optimizer]:
    if model_name == "baseline":
        model = MLP(layer_sizes=[784, 100, 80, 10], dropout_p=0.2)
    elif model_name == "wider_net":
        model = MLP(layer_sizes=[784, 256, 128, 10], dropout_p=0.3)
    elif model_name == "deeper_net":
        model = MLP(layer_sizes=[784, 128, 64, 32, 10], dropout_p=0.2)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)
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
    parser = argparse.ArgumentParser(description="PyTorch MNIST MLP Experiment")
    parser.add_argument("--model_name",type=str,default="baseline",help="模型名称 (baseline, wider_net, deeper_net)")
    parser.add_argument("--optimizer", type=str, default="sgd", help="优化器 (sgd, adam)")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--exp_name",type=str,default="mlp_test",help="实验名称，用于保存图表和模型")
    parser.add_argument("--log_interval", type=int, default=200, help="日志记录间隔")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Running Experiment: {args.exp_name}")
    print(f"Using device: {args.device}")

    train_loader, validation_loader = get_data_loaders(batch_size=args.batch_size)

    model, optimizer = get_model_and_optimizer(args.model_name, args.optimizer, args.lr)
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()

    print("Model Architecture:")
    print(model)
    print("\nOptimizer:")
    print(optimizer)

    lossv, accv = [], []
    for epoch in range(1, args.epochs + 1):
        train_epoch(model,args.device,train_loader,optimizer,criterion,epoch,args.log_interval)
        validate_epoch(model, args.device, validation_loader, criterion, lossv, accv)

    output_dir = f"out/mlp/{args.exp_name}"
    figure_path = f"fig/mlp/{args.exp_name}/mlp_performance_{args.exp_name}.png"
    os.makedirs(output_dir, exist_ok=True)
    
    utils.plot_performance(lossv, accv, save_path=figure_path, title=f"MLP - {args.exp_name}")

    model_save_path = f"{output_dir}/model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main() 