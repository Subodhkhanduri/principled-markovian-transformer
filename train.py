# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from markov_transformer import MarkovianTransformer
from dataset_factory import get_dataset

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'uci_har', 'financial'])
args = parser.parse_args()

# Dataset
train_loader, val_loader, test_loader = get_dataset(args.dataset)

# Determine model input dimensions
sample_input, _ = next(iter(train_loader))
seq_len, input_dim = sample_input.shape[1], sample_input.shape[2]
num_classes = 1 if args.dataset in ['synthetic', 'financial'] else torch.max(_).item() + 1

# Model, loss, optimizer
model = MarkovianTransformer(input_dim=input_dim, hidden_dim=64, n_heads=4)
if num_classes == 1:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_x)

        if num_classes == 1:
            loss = criterion(logits, batch_y)
            predictions = torch.sigmoid(logits).round()
        else:
            batch_y = batch_y.view(-1)
            loss = criterion(logits, batch_y)
            predictions = torch.argmax(logits, dim=1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), f"markovian_transformer_{args.dataset}.pt")
