#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a CNN-based heuristic function for grid path planning.

Assumptions on dataset (*.npz):
    - X: shape (N, 3, H, W), dtype float32
        channel 0: obstacle map (1 for obstacle, 0 for free)
        channel 1: agent position (1 at agent cell, else 0)
        channel 2: goal position (1 at goal cell, else 0)
    - y: shape (N,), shortest path length (int or float)

You (B teammate) should generate:
    dataset_train.npz, dataset_val.npz

Usage:
    python train_heuristic.py \
        --train dataset_train.npz \
        --val dataset_val.npz \
        --epochs 20 \
        --lr 1e-3 \
        --save_model heuristic_cnn.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ====================
# 1. Dataset
# ====================

class GridHeuristicDataset(Dataset):
    """PyTorch Dataset for grid states and shortest distances."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]          # (N, C, H, W)
        self.y = data["y"]          # (N,)
        # convert to float32
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]                    # (C, H, W)
        y = self.y[idx]                    # scalar
        return torch.from_numpy(x), torch.tensor(y)


# ====================
# 2. Model
# ====================

class HeuristicCNN(nn.Module):
    """Simple CNN to predict distance-to-goal from grid state."""

    def __init__(self, in_channels=3, h=15, w=15):
        super().__init__()
        # you can adjust architecture here
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # downsample by 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # compute flattened size after conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            out = self.features(dummy)
            flat_dim = out.view(1, -1).shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)     # predict one scalar: distance
        )

    def forward(self, x):
        # x: (B, C, H, W)
        feat = self.features(x)
        feat = feat.view(x.size(0), -1)
        out = self.regressor(feat)
        return out.squeeze(-1)    # (B,)


# ====================
# 3. Train & Eval
# ====================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X)           # (B,)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

    return total_loss / n_samples


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)
            mae = torch.mean(torch.abs(pred - y))

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples, total_mae / n_samples


# ====================
# 4. Helper: load model & heuristic fn (给 A 同学用)
# ====================

def load_trained_model(model_path, h=15, w=15, device=None):
    """Load a trained heuristic model for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeuristicCNN(in_channels=3, h=h, w=w).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def heuristic_fn(model, device, state_np):
    """
    Compute heuristic for a single state.

    Args:
        model: HeuristicCNN (already in eval mode)
        device: torch.device
        state_np: numpy array of shape (3, H, W), float32

    Returns:
        float: predicted distance
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)  # (1,3,H,W)
        x = x.to(device)
        pred = model(x)  # (1,)
        return float(pred.item())


# ====================
# 5. Main
# ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="dataset_train.npz")
    parser.add_argument("--val", type=str, default="dataset_val.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_model", type=str, default="heuristic_cnn.pt")
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets & loaders
    train_dataset = GridHeuristicDataset(args.train)
    val_dataset = GridHeuristicDataset(args.val)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Build model
    model = HeuristicCNN(
        in_channels=3, h=args.height, w=args.width
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = eval_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
        )

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_model)
            print(f"  -> New best model saved to {args.save_model}")

    print("Training finished. Best val MSE:", best_val_loss)

    # Collect all predictions on validation set
    model.eval()
    all_pred = []
    all_true = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)

            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    # ---------- Histogram of Errors ----------
    errors = all_pred - all_true
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color="skyblue", edgecolor="black")
    plt.title("Prediction Error Histogram (pred - true)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig("error_histogram.png")
    plt.close()

    # ---------- Scatter Plot ----------
    plt.figure(figsize=(8, 6))
    plt.scatter(all_true, all_pred, alpha=0.5, s=10)
    plt.title("Predicted vs True Shortest Distance")
    plt.xlabel("True Distance")
    plt.ylabel("Predicted Distance")
    plt.grid(alpha=0.3)
    # Draw y=x reference line
    min_v = min(all_true.min(), all_pred.min())
    max_v = max(all_true.max(), all_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--")
    plt.savefig("scatter_pred_vs_true.png")
    plt.close()

    print("Saved evaluation plots: error_histogram.png, scatter_pred_vs_true.png")


if __name__ == "__main__":
    main()
