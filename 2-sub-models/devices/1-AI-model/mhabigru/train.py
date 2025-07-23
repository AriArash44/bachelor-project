import argparse
import pickle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from model import MHABiGRU

class TabularSeqDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series = None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.X = self.X.unsqueeze(-1)
        self.y = None
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

def train(args):
    df = pd.read_csv(args.train_csv)
    X = df.drop(columns=["type"])
    y = df["type"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    ds = TabularSeqDataset(X, pd.Series(y_enc))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    seq_len = X.shape[1]
    input_size = 1
    model = MHABiGRU(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_classes=num_classes,
        dropout=args.dropout
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        avg = total_loss / len(ds)
        print(f"Epoch {epoch}/{args.epochs} loss={avg:.4f}")
    torch.save(model.state_dict(), args.model_out)
    with open(args.label_map, "wb") as f:
        pickle.dump(le, f)
    print("Training complete.")
    print(f"→ Model saved to: {args.model_out}")
    print(f"→ Label encoder saved to: {args.label_map}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train MHABiGRU on tabular‐as‐seq")
    p.add_argument("--train-csv", required=True, help="preprocessed.csv (with 'y' col)")
    p.add_argument("--model-out", default="mhabigru_model.pt", help="where to save model")
    p.add_argument("--label-map", default="label_map.pkl", help="LabelEncoder pickle")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    train(args)
