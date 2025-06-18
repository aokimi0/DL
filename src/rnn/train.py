import argparse
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split

from .. import utils
from .data import (
    N_LETTERS,
    SurnameDataset,
    category_from_output,
    collate_fn,
    get_random_training_example,
    load_surname_data_for_manual_models,
)
from .models.gru import ManualGRU
from .models.lstm import ManualLSTM
from .models.manual_rnn import ManualRNN
from .models.nn_gru import NNGRU
from .models.nn_lstm import NNLSTM
from .models.nn_rnn import NNRNN


def get_model(
    model_name: str, n_letters: int, n_hidden: int, n_categories: int
) -> nn.Module:
    """Factory function to create and return a model based on its name."""
    models = {
        "manual_rnn": ManualRNN,
        "manual_lstm": ManualLSTM,
        "manual_gru": ManualGRU,
        "nn_rnn": NNRNN,
        "nn_lstm": NNLSTM,
        "nn_gru": NNGRU,
    }
    model_class = models.get(model_name)
    if model_class:
        return model_class(n_letters, n_hidden, n_categories)
    raise ValueError(f"Unknown model name: {model_name}")


def evaluate_nn_model(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Evaluate a batch-based nn model."""
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for lines, categories, _ in loader:
            lines, categories = lines.to(device), categories.to(device)
            hidden = model.init_hidden(batch_size=lines.size(0))
            output, _ = model(lines, hidden)
            loss = criterion(output, categories)
            
            total_loss += loss.item() * lines.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == categories).sum().item()
            total_samples += lines.size(0)
            
    model.train()
    return total_loss / total_samples, (correct / total_samples) * 100


def evaluate_manual_model(
    model: nn.Module, criterion: nn.Module, all_categories: List[str], 
    category_lines: dict, device: torch.device, n_samples: int = 1000
) -> Tuple[float, float]:
    """Evaluate a manual, single-instance model."""
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for _ in range(n_samples):
            _, _, category_tensor, line_tensor = get_random_training_example(all_categories, category_lines)
            line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)
            
            hidden = model.init_hidden().to(device)
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)

            for i in range(line_tensor.size(0)):
                output, hidden = model(line_tensor[i], hidden)
            
            loss = criterion(output, category_tensor)
            total_loss += loss.item()
            
            guess, _ = category_from_output(output, all_categories)
            if all_categories[category_tensor.item()] == guess:
                correct += 1
                
    model.train()
    return total_loss / n_samples, (correct / n_samples) * 100


def train_nn_model(args, model, train_loader, val_loader, criterion, optimizer, device, all_categories):
    """Main training loop for batch-based nn models."""
    print(f"\nTraining model: {args.model.upper()} on {device}")
    print(model)
    
    all_losses, all_accuracies = [], []
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        for i, (lines, categories, _) in enumerate(train_loader):
            lines, categories = lines.to(device), categories.to(device)
            
            optimizer.zero_grad()
            hidden = model.init_hidden(batch_size=lines.size(0))
            output, _ = model(lines, hidden)
            loss = criterion(output, categories)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            
            epoch_loss += loss.item()

        val_loss, val_acc = evaluate_nn_model(model, val_loader, criterion, device)
        all_losses.append(val_loss)
        all_accuracies.append(val_acc)
        
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Time: {time.time() - start_time:.0f}s | "
            f"Train Loss: {epoch_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

    return all_losses, all_accuracies


def train_manual_model(args, model, criterion, optimizer, device, all_categories, category_lines):
    """Training loop for manual, single-instance models."""
    print(f"\nTraining model: {args.model.upper()} with gradient accumulation on {device}")
    print(model)
    
    all_losses, all_accuracies = [], []
    current_train_loss = 0
    start_time = time.time()
    
    optimizer.zero_grad()
    for i in range(1, args.n_iters + 1):
        category, line, category_tensor, line_tensor = get_random_training_example(all_categories, category_lines)
        line_tensor, category_tensor = line_tensor.to(device), category_tensor.to(device)
        
        hidden = model.init_hidden()
        if isinstance(hidden, tuple):
            hidden = tuple(h.to(device) for h in hidden)
        else:
            hidden = hidden.to(device)

        for j in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[j], hidden)
        
        loss = criterion(output, category_tensor)
        (loss / args.batch_size).backward()
        current_train_loss += loss.item()

        if i % args.batch_size == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()

        if i % args.print_every == 0:
            guess, _ = category_from_output(output, all_categories)
            correct = "✓" if guess == category else f"✗ ({category})"
            print(f"{i:>6d} {i/args.n_iters*100:>3.0f}% ({time.time()-start_time:.0f}s) Loss: {loss.item():.4f} | {line} -> {guess} {correct}")

        if i % args.plot_every == 0:
            val_loss, val_acc = evaluate_manual_model(model, criterion, all_categories, category_lines, device)
            train_loss = current_train_loss / args.plot_every
            all_losses.append(val_loss)
            all_accuracies.append(val_acc)
            current_train_loss = 0
            print(f"Validation @ {i}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    return all_losses, all_accuracies


def main():
    parser = argparse.ArgumentParser(
        description="Train RNN models for surname classification."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "manual_rnn",
            "manual_lstm",
            "manual_gru",
            "nn_rnn",
            "nn_lstm",
            "nn_gru",
        ],
        help="Model to train.",
    )
    parser.add_argument(
        "--n_iters", type=int, default=100000, help="Number of training samples to process."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for gradient accumulation."
    )
    parser.add_argument(
        "--n_hidden", type=int, default=128, help="Size of the hidden layer."
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument(
        "--print_every", type=int, default=5000, help="Log frequency (in iterations)."
    )
    parser.add_argument(
        "--plot_every", type=int, default=1000, help="Plotting data frequency (in iterations)."
    )
    parser.add_argument(
        "--exp_name", type=str, required=True, help="Experiment name for outputs."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Disable CUDA."
    )
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    if "manual" in args.model:
        category_lines, all_categories = load_surname_data_for_manual_models()
        n_categories = len(all_categories)
        model = get_model(args.model, N_LETTERS, args.n_hidden, n_categories).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.NLLLoss()
        all_losses, all_accuracies = train_manual_model(args, model, criterion, optimizer, device, all_categories, category_lines)
    else:
        dataset = SurnameDataset()
        n_categories = dataset.n_categories
        all_categories = dataset.all_categories
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        model = get_model(args.model, N_LETTERS, args.n_hidden, n_categories).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.NLLLoss()
        all_losses, all_accuracies = train_nn_model(args, model, train_loader, val_loader, criterion, optimizer, device, all_categories)

    print("\nGenerating final evaluation plots...")
    output_dir = f"fig/rnn/{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    title = f"Performance: {args.model.replace('_', ' ').title()}"
    utils.plot_performance(
        all_losses,
        all_accuracies,
        save_path=f"{output_dir}/performance_{args.exp_name}.png",
        title=title,
    )
    
    if "manual" in args.model:
        confusion = torch.zeros(n_categories, n_categories)
        model.eval()
        for _ in range(10000):
            category, _, category_tensor, line_tensor = get_random_training_example(
                all_categories, category_lines
            )
            
            device = next(model.parameters()).device
            line_tensor = line_tensor.to(device)
            hidden = model.init_hidden()
            if isinstance(hidden, tuple):
                hidden = tuple(h.to(device) for h in hidden)

            for j in range(line_tensor.size(0)):
                output, hidden = model(line_tensor[j], hidden)
            
            _, guess_i = category_from_output(output, all_categories)
            confusion[category_tensor.item()][guess_i] += 1
            
        utils.plot_confusion_matrix(
            confusion,
            all_categories,
            save_path=f"{output_dir}/confusion_matrix.png",
            title=f"Confusion Matrix: {args.exp_name}",
            normalize=True,
        )
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main() 