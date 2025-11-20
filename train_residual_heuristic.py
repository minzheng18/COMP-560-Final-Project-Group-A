#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class GridHeuristicDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32)  # (N,3,H,W)
        self.y = data["y"].astype(np.float32)  # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])


class HeuristicCNN(nn.Module):
    def __init__(self, in_channels=3, h=15, w=15):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),               # H,W -> H/2, W/2

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),               # H/2,W/2 -> H/4, W/4
        )

        # compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            flat_dim = self.features(dummy).view(1, -1).shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.regressor(feat).squeeze(-1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n += X.size(0)

    return total_loss / n


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    n = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            loss = criterion(pred, y)
            mae = torch.mean(torch.abs(pred - y))

            total_loss += loss.item() * X.size(0)
            total_mae += mae.item() * X.size(0)
            n += X.size(0)

    return total_loss / n, total_mae / n



def load_trained_model(model_path, h=15, w=15, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeuristicCNN(3, h, w).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def heuristic_fn(model, device, state_np):
    with torch.no_grad():
        x = torch.from_numpy(state_np.astype(np.float32)).unsqueeze(0)
        x = x.to(device)
        return float(model(x).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="dataset_train_residual.npz")
    parser.add_argument("--val",   type=str, default="dataset_val_residual.npz")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--save_model", type=str, default="heuristic.pt")
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width",  type=int, default=15)
    parser.add_argument("--residual", action="store_true",
                        help="If set, train NN to predict residual (dist - manhattan)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    train_set = GridHeuristicDataset(args.train)
    val_set   = GridHeuristicDataset(args.val)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch, shuffle=False)

    model = HeuristicCNN(3, args.height, args.width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = eval_model(model, val_loader, criterion, device)

        print(f"[Epoch {epoch:02d}] Train={train_loss:.4f} | Val={val_loss:.4f} | MAE={val_mae:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.save_model)
            print("  -> Saved best model to", args.save_model)


    print("\nGenerating evaluation plots ...")

    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            pred = model(X)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(all_pred - all_true, bins=40, edgecolor="black")
    plt.title("Prediction Error Histogram (pred - true)")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.savefig("error_histogram_residual.png")
    plt.close()

    # Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(all_true, all_pred, alpha=0.4, s=10)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Predicted vs True")
    min_v = min(all_true.min(), all_pred.min())
    max_v = max(all_true.max(), all_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--")
    plt.savefig("scatter_pred_vs_true_residual.png")
    plt.close()

    print("Saved: error_histogram_residual.png, scatter_pred_vs_true_residual.png")


if __name__ == "__main__":
    main()
