import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_columns] = scaler.fit_transform(df_scaled[numeric_columns])
    return scaler, numeric_columns, df_scaled

def fit_pipeline(X: pd.DataFrame):
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    X_scaled = X.copy()
    X_scaled[numeric_columns] = scaler.fit_transform(X_scaled[numeric_columns])
    final_columns = X_scaled.columns.tolist()
    return scaler, numeric_columns, final_columns, X_scaled

def transform_pipeline(
    X: pd.DataFrame,
    scaler: MinMaxScaler,
    numeric_columns: list,
    final_columns: list
) -> pd.DataFrame:
    X_scaled = X.copy()
    missing = [col for col in numeric_columns if col not in X.columns]
    if missing:
        raise KeyError(f"Missing expected numeric columns: {missing}")
    X_scaled[numeric_columns] = scaler.transform(X_scaled[numeric_columns])
    return X_scaled[final_columns]


def main():
    p = argparse.ArgumentParser(
        description="Fit or apply a 0–1 minmax scaler to a CSV dataset"
    )
    sub = p.add_subparsers(dest="mode", required=True)
    fit = sub.add_parser("fit", help="Fit scaler on train set and save artifacts")
    fit.add_argument("--train-csv", required=True, help="Input CSV for fitting")
    fit.add_argument("--out-x-csv", required=True, help="Where to write scaled X")
    fit.add_argument("--out-y-csv", required=True, help="Where to write y column")
    fit.add_argument("--preproc-pkl", default="normalize.pkl",
                     help="Pickle file to save scaler + metadata")
    tr = sub.add_parser("transform", help="Load scaler and apply to new CSV")
    tr.add_argument("--in-csv", required=True, help="Input CSV to transform")
    tr.add_argument("--out-x-csv", required=True, help="Where to write scaled X")
    tr.add_argument("--out-y-csv", default=None, help="Where to write y column if present")
    tr.add_argument("--preproc-pkl", default="normalize.pkl",
                    help="Pickle file with fitted scaler + metadata")
    args = p.parse_args()
    if args.mode == "fit":
        df = pd.read_csv(args.train_csv)
        y  = df.pop("type")
        X  = df
        scaler, nums, cols, X_scaled = fit_pipeline(X)
        X_scaled.to_csv(args.out_x_csv, index=False)
        pd.DataFrame({"type": y}).to_csv(args.out_y_csv, index=False)
        with open(args.preproc_pkl, "wb") as f:
            pickle.dump({
                "scaler":            scaler,
                "numeric_columns":   nums,
                "final_columns":     cols
            }, f)
        print(f"Fitted scaler saved to {args.preproc_pkl}")
    else:
        df = pd.read_csv(args.in_csv)
        has_y = "type" in df.columns
        y = df.pop("type") if has_y else None
        X = df
        meta = pickle.load(open(args.preproc_pkl, "rb"))
        X_scaled = transform_pipeline(
            X,
            meta["scaler"],
            meta["numeric_columns"],
            meta["final_columns"]
        )
        X_scaled.to_csv(args.out_x_csv, index=False)
        if has_y and args.out_y_csv:
            pd.DataFrame({"type": y}).to_csv(args.out_y_csv, index=False)
        print(f"Transformed data written to {args.out_x_csv}")

if __name__ == "__main__":
    main()
