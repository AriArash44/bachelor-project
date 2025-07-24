import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from model import build_model

def slide_windows(X: np.ndarray, y: np.ndarray, seq_len: int):
    N, F = X.shape
    n_blocks = N // seq_len
    W = X[: n_blocks * seq_len].reshape(n_blocks, seq_len, F)
    L = y[seq_len - 1 : n_blocks * seq_len : seq_len]
    return W, L

def compute_class_weights(labels: np.ndarray) -> dict:
    counts = np.bincount(labels, minlength=labels.max() + 1)
    total = counts.sum()
    num_classes = len(counts)
    return {i: total / (num_classes * c) for i, c in enumerate(counts)}

def train(args):
    df = pd.read_csv(args.train_csv)
    X_raw = df.drop(columns=["type"]).values.astype("float32")
    y_raw = LabelEncoder().fit_transform(df["type"].values)
    W, L = slide_windows(X_raw, y_raw, args.seq_len)
    class_weights = compute_class_weights(L)
    model = build_model(
        seq_len=args.seq_len,
        features=X_raw.shape[1],
        hidden_size=args.hidden_size,
        num_classes=len(np.unique(L)),
        dropout=args.dropout,
        num_heads=args.num_heads
    )
    model.compile(
        optimizer=Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        W, L,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        class_weight=class_weights,
        verbose=1
    )
    model.save(args.model_out, include_optimizer=False)
    with open(args.label_map, "wb") as f:
        pickle.dump(LabelEncoder().fit(df["type"]), f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--model-out", default="model_tf.h5")
    p.add_argument("--label-map", default="label_map.pkl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--use-attention", type=bool, default=True)
    p.add_argument("--num-heads", type=int, default=2)
    args = p.parse_args()
    train(args)
