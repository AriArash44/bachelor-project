import sys
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

def normalize(input_csv: str,
              normalizer_script: str,
              normalizer_pkl: str,
              temp_dir: Path) -> Path:
    out_x = temp_dir / "0-X_norm.csv"
    cmd = [
        sys.executable, normalizer_script,
        "transform",
        "--in-csv", str(input_csv),
        "--out-x-csv", str(out_x),
        "--preproc-pkl", normalizer_pkl,
    ]
    print(">> Normalizing:\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_x

def feature_select(norm_csv: Path,
                   selector_script: str,
                   selector_pkl: str,
                   temp_dir: Path) -> Path:
    out_x = temp_dir / "1-X_preprocessed.csv"
    cmd = [
        sys.executable, selector_script,
        "transform",
        "--in-csv", str(norm_csv),
        "--out-x-csv", str(out_x),
        "--preproc-pkl", selector_pkl,
    ]
    print(">> Feature selecting:\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_x

def slide_windows(X: np.ndarray, seq_len: int) -> np.ndarray:
    N, F = X.shape
    n_blocks = N // seq_len
    return X[: n_blocks * seq_len].reshape(n_blocks, seq_len, F)

def predict_direct(selected_csv: Path,
                   model_h5: str,
                   label_map: str,
                   seq_len: int,
                   batch_size: int,
                   temp_dir: Path) -> Path:
    out_y = temp_dir / "2-y_pred.csv"
    df = pd.read_csv(selected_csv)
    X = df.values.astype("float32")
    W = slide_windows(X, seq_len)
    model = load_model(model_h5)
    with open(label_map, "rb") as f:
        label_encoder = pickle.load(f)
    preds = model.predict(W, batch_size=batch_size)
    class_idxs = preds.argmax(axis=1)
    labels = label_encoder.inverse_transform(class_idxs)
    probs_df = pd.DataFrame(preds, columns=label_encoder.classes_)
    probs_df["predicted"] = labels
    probs_df.to_csv(out_y, index=False)
    print(f">> Predictions saved to: {out_y}")
    return out_y

def driver_main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Normalize → Feature-select → Window → Predict"
    )
    parser.add_argument("input_csv", help="Raw features CSV")
    parser.add_argument("--normalizer-script",
                        default="../0-preprocessors/2-normalizer/normalizer.py")
    parser.add_argument("--normalizer-pkl",
                        default="../0-preprocessors/2-normalizer/normalize.pkl")
    parser.add_argument("--selector-script",
                        default="../0-preprocessors/3-featureSelector/featureSelector.py")
    parser.add_argument("--selector-pkl",
                        default="../0-preprocessors/3-featureSelector/feature_selection.pkl")
    parser.add_argument("--model-h5",
                        default="../1-AI-model/mhabigru/model_tf.h5")
    parser.add_argument("--label-map",
                        default="../1-AI-model/mhabigru/label_map.pkl")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(cli_args)
    script_dir = Path(__file__).resolve().parent
    temp_dir = script_dir / "temp_files"
    temp_dir.mkdir(exist_ok=True)
    norm_path = normalize(
        input_csv=args.input_csv,
        normalizer_script=args.normalizer_script,
        normalizer_pkl=args.normalizer_pkl,
        temp_dir=temp_dir,
    )
    sel_path = feature_select(
        norm_csv=norm_path,
        selector_script=args.selector_script,
        selector_pkl=args.selector_pkl,
        temp_dir=temp_dir,
    )
    predict_direct(
        selected_csv=sel_path,
        model_h5=args.model_h5,
        label_map=args.label_map,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        temp_dir=temp_dir,
    )

if __name__ == "__main__":
    driver_main()