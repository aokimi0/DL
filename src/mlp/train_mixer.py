import torch
import torch.optim as optim
import torch.nn as nn
from typing import List
import os
import argparse

from .mixer_model import MlpMixer
from .data_loader import get_data_loaders
from .. import utils

def get_args():
    parser = argparse.ArgumentParser(description='Train an MLP-Mixer model on MNIST.')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='AdamW weight decay (default: 1e-4)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--exp_name', type=str, default='baseline', help='name for the experiment, used for output directories')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    
    return parser.parse_args()

def train_epoch(model: nn.Module, device: torch.device, train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, epoch: int, log_interval: int):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def validate_epoch(model: nn.Module, device: torch.device, validation_loader: torch.utils.data.DataLoader, criterion: nn.Module, loss_vector: List[float], accuracy_vector: List[float]):
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)
    
    accuracy = 100. * correct / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(validation_loader.dataset)} ({accuracy:.0f}%)\n')

def main():
    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print("Running MLP-Mixer Experiment")
    print(f"Using device: {device}")
    print(f"Experiment Name: {args.exp_name}")

    train_loader, validation_loader = get_data_loaders(batch_size=args.batch_size)

    model = MlpMixer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print("Model Architecture:")
    print(model)

    lossv, accv = [], []
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, device, train_loader, optimizer, criterion, epoch, args.log_interval)
        validate_epoch(model, device, validation_loader, criterion, lossv, accv)
    
    exp_dir_name = f"mixer_{args.exp_name}"
    output_dir = f"out/mlp/{exp_dir_name}"
    figures_dir = f"fig/mlp/{exp_dir_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    figure_save_path = f"{figures_dir}/performance_{exp_dir_name}.png"
    utils.plot_performance(lossv, accv, title=f"MLP-Mixer 性能 ({exp_dir_name})", save_path=figure_save_path)

    model_save_path = f"{output_dir}/mixer_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == '__main__':
    main() 