import argparse
import os
import time
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import SGD

from .. import utils
from .data import (
    N_LETTERS,
    get_random_training_example,
    line_to_tensor,
    load_surname_data,
)
from .models.lstm import LSTM
from .models.manual_rnn import ManualRNN


def get_model(
    model_name: str, n_letters: int, n_hidden: int, n_categories: int
) -> nn.Module:
    if model_name == "rnn":
        return ManualRNN(n_letters, n_hidden, n_categories)
    if model_name == "lstm":
        return LSTM(n_letters, n_hidden, n_categories)
    raise ValueError(f"Unknown model name: {model_name}")


def train_step(
    model: nn.Module,
    category_tensor: torch.Tensor,
    line_tensor: torch.Tensor,
    optimizer: SGD,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, float]:
    optimizer.zero_grad()
    
    device = next(model.parameters()).device
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    
    hidden = model.init_hidden()
    if isinstance(hidden, tuple):
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = hidden.to(device)

    if isinstance(model, ManualRNN):
        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)
        final_output = output
    else:
        output, hidden = model(line_tensor, hidden)
        final_output = output[-1]

    loss = criterion(final_output, category_tensor)
    loss.backward()
    optimizer.step()
    
    return final_output, loss.item()


def evaluate(model: nn.Module, line_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        device = next(model.parameters()).device
        line_tensor = line_tensor.to(device)
        hidden = model.init_hidden()
        if isinstance(hidden, tuple):
             hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)

        if isinstance(model, ManualRNN):
            for i in range(line_tensor.size()[0]):
                output, hidden = model(line_tensor[i], hidden)
            final_output = output
        else:
            output, hidden = model(line_tensor, hidden)
            final_output = output[-1]
            
        return final_output


def category_from_output(output: torch.Tensor, all_categories: List[str]) -> Tuple[str, int]:
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def main():
    parser = argparse.ArgumentParser(description="Train RNN models for surname classification.")
    parser.add_argument("--model", type=str, required=True, choices=["rnn", "lstm"], help="Model to train.")
    parser.add_argument("--n_iters", type=int, default=100000, help="Number of training iterations.")
    parser.add_argument("--n_hidden", type=int, default=128, help="Size of the hidden layer.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--print_every", type=int, default=5000, help="Log frequency.")
    parser.add_argument("--plot_every", type=int, default=1000, help="Plotting data frequency.")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for outputs.")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="Disable CUDA.")
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    category_lines, all_categories = load_surname_data()
    n_categories = len(all_categories)

    model = get_model(args.model, N_LETTERS, args.n_hidden, n_categories).to(device)
    optimizer = SGD(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    print(f"\nTraining model: {args.model.upper()}")
    print(model)

    all_losses = []
    current_loss = 0
    start_time = time.time()

    for i in range(1, args.n_iters + 1):
        category, line, category_tensor, line_tensor = get_random_training_example(all_categories, category_lines)
        output, loss = train_step(model, category_tensor, line_tensor, optimizer, criterion)
        current_loss += loss

        if i % args.print_every == 0:
            guess, _ = category_from_output(output, all_categories)
            correct = "✓" if guess == category else f"✗ ({category})"
            elapsed = time.time() - start_time
            print(f"{i:>6d} {i / args.n_iters * 100:>3.0f}% ({elapsed:.0f}s) Loss: {loss:.4f} | {line} -> {guess} {correct}")

        if i % args.plot_every == 0:
            all_losses.append(current_loss / args.plot_every)
            current_loss = 0

    print("\nGenerating final evaluation plots...")
    output_dir = f"fig/rnn/{args.exp_name}"
    os.makedirs(output_dir, exist_ok=True)

    utils.plot_loss(all_losses, save_path=f"{output_dir}/loss.png")

    confusion = torch.zeros(n_categories, n_categories)
    for _ in range(10000):
        category, _, _, line_tensor = get_random_training_example(all_categories, category_lines)
        output = evaluate(model, line_tensor)
        guess, guess_i = category_from_output(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1
    
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
    
    utils.plot_confusion_matrix(confusion.numpy(), all_categories, save_path=f"{output_dir}/confusion_matrix.png")
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main() 