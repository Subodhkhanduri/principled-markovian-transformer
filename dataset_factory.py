# dataset_factory.py

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def generate_synthetic_dataset(seq_len=50, num_samples=1000, input_dim=16):
    """Generate synthetic autoregressive + chaotic sequences."""
    X = []
    y = []
    for _ in range(num_samples):
        init = np.random.randn(input_dim)
        sequence = [init]
        for _ in range(seq_len - 1):
            prev = sequence[-1]
            # AR + logistic map combo
            noise = np.random.normal(0, 0.1, size=input_dim)
            next_step = 0.7 * prev + 0.3 * (4 * prev * (1 - prev)) + noise
            sequence.append(next_step)
        seq = np.stack(sequence)
        label = (seq.mean() > 0).astype(np.float32)
        X.append(seq)
        y.append(label)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X, y)


def load_uci_har(path='data/UCI_HAR/processed_data.pt'):
    """Load preprocessed UCI HAR data."""
    try:
        data = torch.load(path)
        X = data['X']  # shape: (N, T, D)
        y = data['y'].long()
    except FileNotFoundError:
        raise RuntimeError("Preprocessed UCI HAR not found. Please preprocess first.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    return train_loader, val_loader, int(y.max().item()) + 1


def load_financial_dummy(seq_len=50, num_samples=1000, input_dim=16):
    """Dummy financial sequence data mimicking stock trends."""
    X = []
    y = []
    for _ in range(num_samples):
        trend = np.cumsum(np.random.randn(seq_len, input_dim) * 0.05, axis=0)
        noise = np.random.randn(seq_len, input_dim) * 0.02
        sequence = trend + noise
        label = (trend[-1].mean() > trend[0].mean()).astype(np.float32)  # Uptrend = 1
        X.append(sequence)
        y.append(label)
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X, y)


def get_dataset(name, batch_size=32):
    """
    name: one of ["synthetic", "uci_har", "financial"]
    returns: train_loader, val_loader, num_classes (1 for binary)
    """
    if name == "synthetic":
        dataset = generate_synthetic_dataset()
        train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size), 1)

    elif name == "uci_har":
        return load_uci_har()

    elif name == "financial":
        dataset = load_financial_dummy()
        train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])
        return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                DataLoader(val_ds, batch_size=batch_size), 1)

    else:
        raise ValueError(f"Unknown dataset name: {name}")