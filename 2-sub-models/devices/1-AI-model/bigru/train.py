import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from model import build_bigru

def create_context_windows(X, y, context):
    X_windows, y_labels = [], []
    for i in range(context, len(X) - context):
        window = X[i - context : i + context + 1]
        X_windows.append(window)
        y_labels.append(y[i])
    return np.array(X_windows), np.array(y_labels)

def compute_class_weights(labels: np.ndarray) -> dict:
    counts = np.bincount(labels, minlength=labels.max() + 1)
    total = counts.sum()
    num_classes = len(counts)
    return {i: total / (num_classes * c) for i, c in enumerate(counts)}

def train(args):
    df = pd.read_csv(args.train_csv)
    X_raw = df.drop(columns=["type"]).values.astype("float32")
    label_encoder = LabelEncoder()
    y_raw = label_encoder.fit_transform(df["type"].values)

    W, L = create_context_windows(X_raw, y_raw, context=args.context)
    class_weights = compute_class_weights(L)

    model = build_bigru(
        seq_len=W.shape[1],
        features=W.shape[2],
        hidden_size=args.hidden_size,
        num_classes=len(np.unique(L)),
        dropout=args.dropout
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
        pickle.dump(label_encoder, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--model-out", default="bigru_tf.h5")
    p.add_argument("--label-map", default="label_map.pkl")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--context", type=int, default=10)
    args = p.parse_args()
    train(args)
