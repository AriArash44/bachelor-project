import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from model import MHABiGRU

class SequenceDataset(Dataset):
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 seq_len: int):
        self.seq_len = seq_len
        self.features = X.values.astype(np.float32)
        self.labels = y.values.astype(np.int64)
        self.n_samples = len(self.features) - seq_len + 1

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        window = self.features[start:end]
        label = self.labels[end - 1]
        return torch.from_numpy(window), torch.tensor(label)

def train(args):
    df = pd.read_csv(args.train_csv)
    X = df.drop(columns=["type"])
    y = df["type"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    dataset = SequenceDataset(X, pd.Series(y_enc), seq_len=args.seq_len)
    class_counts = np.bincount(dataset.labels, minlength=len(le.classes_))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MHABiGRU(
        input_size=X.shape[1],
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_classes=len(le.classes_),
        dropout=args.dropout
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{args.epochs} – loss={avg_loss:.4f}")
    torch.save(model.state_dict(), args.model_out)
    with open(args.label_map, "wb") as f:
        pickle.dump(le, f)
    print("🛠 Training complete.")
    print(f"→ Model checkpoint: {args.model_out}")
    print(f"→ Label encoder: {args.label_map}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train MHABiGRU with sequences & imbalance fix")
    p.add_argument("--train-csv", required=True, help="preprocessed.csv (with 'type' column)")
    p.add_argument("--model-out", default="mhabigru_model.pt")
    p.add_argument("--label-map", default="label_map.pkl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=20,
                   help="Number of timesteps in each input window")
    args = p.parse_args()
    train(args)
