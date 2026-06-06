#!/usr/bin/env python3
"""
Simple fine-tuning script for RefineCAM models on synthetic_all dataset.
Takes model name as input, uses synthetic_all dataset for training.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Import models
from models import (
    vgg11_Synthetic,
    vgg_preprocess,
    resnet18_Synthetic,
    resnet50_Synthetic,
    resnet_preprocess,
    swin_Synthetic,
    swin_preprocess,
    vit_Synthetic,
    vit_preprocess,
    convnext_tiny_Synthetic,
    convnext_small_Synthetic,
    convnext_preprocess,
)

# Import dataset
from data import SyntheticFiguresAll

# Dataset root paths - can be overridden via environment variables
WALDO_TRAIN = "./data/WaldoNoise_train"
WALDO_VAL = "./data/WaldoNoise_val"


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(model, dl_train, dl_val, device, epochs, lr=0.0001, model_name="model"):
    """Training loop with validation and early stopping."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(
        patience=7, verbose=True, path=f"checkpoint_{model_name}.pt"
    )
    logger = SummaryWriter(log_dir=f"runs/{model_name}")

    train_history = {"accuracy": [], "loss": []}
    val_history = {"accuracy": [], "loss": []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for j, item in enumerate(
            tqdm(dl_train, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        ):
            if len(item) == 2:
                images, labels = item
            else:
                images, masks, labels = item
            images = images.to(device)
            labels = labels.to(device).view(-1)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log batch-level metrics
            step = epoch * len(dl_train) + j
            logger.add_scalar("Loss/train_batch", loss.item(), step)

        # Calculate epoch metrics
        train_loss /= len(dl_train)
        train_acc = correct / total
        train_history["loss"].append(train_loss)
        train_history["accuracy"].append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_losses = []

        with torch.no_grad():
            for item in tqdm(dl_val, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                if len(item) == 2:
                    images, labels = item
                else:
                    images, masks, labels = item
                images = images.to(device)
                labels = labels.to(device).view(-1)

                output = model(images)
                loss = criterion(output, labels)
                val_losses.append(loss.item())

                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = sum(val_losses) / len(val_losses)
        val_acc = correct / total
        val_history["loss"].append(val_loss)
        val_history["accuracy"].append(val_acc)

        # Logging
        logger.add_scalar("Loss/train_epoch", train_loss, epoch)
        logger.add_scalar("Loss/val_epoch", val_loss, epoch)
        logger.add_scalar("Accuracy/train_epoch", train_acc, epoch)
        logger.add_scalar("Accuracy/val_epoch", val_acc, epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    logger.close()
    return train_history, val_history


def get_model_and_preprocess(model_name):
    """Get model and preprocessing function based on model name."""
    model_map = {
        "vgg11": (vgg11_Synthetic, vgg_preprocess),
        "resnet18": (resnet18_Synthetic, resnet_preprocess),
        "resnet50": (resnet50_Synthetic, resnet_preprocess),
        "swin": (swin_Synthetic, swin_preprocess),
        "vit": (vit_Synthetic, vit_preprocess),
        "convnext_tiny": (convnext_tiny_Synthetic, convnext_preprocess),
        "convnext_small": (convnext_small_Synthetic, convnext_preprocess),
    }

    if model_name not in model_map:
        raise ValueError(
            f"Unsupported model: {model_name}. Available: {list(model_map.keys())}"
        )

    return model_map[model_name]


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RefineCAM models on synthetic_all dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "vgg11",
            "resnet18",
            "resnet50",
            "swin",
            "vit",
            "convnext_tiny",
            "convnext_small",
        ],
        help="Model architecture to fine-tune",
    )
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--train_size", type=int, default=4096, help="Number of training samples"
    )
    parser.add_argument(
        "--val_size", type=int, default=64, help="Number of validation samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/weights",
        help="Directory to save model weights",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for the experiment (used for logging and checkpoint naming)",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_synthetic"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./runs", exist_ok=True)

    print(f"Fine-tuning {args.model} on synthetic dataset")
    print(f"Experiment name: {args.experiment_name}")

    # Get model and preprocessing function
    model_fn, preprocess_fn = get_model_and_preprocess(args.model)

    # Load datasets
    print("Loading datasets...")
    train_dataset = SyntheticFiguresAll(
        background_path=WALDO_TRAIN,
        num_images=args.train_size,
        split="train",
        image_transform=preprocess_fn,
    )

    val_dataset = SyntheticFiguresAll(
        background_path=WALDO_VAL,
        num_images=args.val_size,
        split="val",
        image_transform=preprocess_fn,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    print("Initializing model...")
    model = model_fn()
    model = model.to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Train model
    print("Starting training...")
    train_history, val_history = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        model_name=args.experiment_name,
    )
    # Save history
    history_path = os.path.join(args.output_dir, f"{args.experiment_name}_history.pt")
    torch.save(
        {"train_history": train_history, "val_history": val_history}, history_path
    )
    print(f"Training history saved to: {history_path}")

    # Save final model
    output_path = os.path.join(args.output_dir, f"{args.experiment_name}.pth")
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to: {output_path}")

    # Also save as .pth.tar for compatibility with existing configs
    output_path_tar = os.path.join(args.output_dir, f"{args.experiment_name}.pth.tar")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
        },
        output_path_tar,
    )
    print(f"Model also saved to: {output_path_tar}")

    print("Training completed!")


if __name__ == "__main__":
    main()
