#!/usr/bin/env python3
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# treat these prefixes like your “devices”
PREFIXES      = ["disk", "memory", "process"]
# additionally scale these two columns
EXTRA_NUMERIC = ["PID", "CMD"]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop the unused label
    if "label" in df.columns:
        df.drop("label", axis=1, inplace=True)

    # build is_off flags for each prefix
    for pre in PREFIXES:
        cols = [c for c in df.columns if c.startswith(pre + ".")]
        df[f"{pre}_is_off"] = df[cols].isnull().all(axis=1).astype(int)

    # impute missing for prefix-based numeric cols
    prefixed_nums = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c.split(".")[0] in PREFIXES
    ]
    for c in prefixed_nums:
        pre = c.split(".")[0]
        off = df[f"{pre}_is_off"] == 1
        on  = ~off
        mn  = df.loc[on, c].min()
        mv  = df.loc[on, c].mean() or 0.0
        df.loc[on,  c] = df.loc[on,  c].fillna(mv)
        df.loc[off, c] = df.loc[off, c].fillna(mn - 1)

    # global-mean impute for PID/CMD if they exist
    for col in EXTRA_NUMERIC:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())

    return df

def fit_pipeline(X: pd.DataFrame):
    # 1) normalize & impute
    Xn = normalize(X)

    # 2) identify numeric columns to scale
    nums = [
        c for c in Xn.select_dtypes(include=[np.number]).columns
        if c.split(".")[0] in PREFIXES or c in EXTRA_NUMERIC
    ]

    # 3) fit & apply MinMaxScaler
    scaler = MinMaxScaler()
    Xn[nums] = scaler.fit_transform(Xn[nums])

    # 4) one-hot encode only the categorical columns
    cat_cols = Xn.select_dtypes(include=["object", "bool", "category"]).columns
    Xe = pd.get_dummies(Xn, columns=cat_cols, drop_first=False, dtype=int)

    return scaler, nums, Xe.columns.tolist(), Xe

def transform_pipeline(X: pd.DataFrame, scaler, nums, final_cols):
    Xn = normalize(X)
    Xn[nums] = scaler.transform(Xn[nums])
    cat_cols = Xn.select_dtypes(include=["object", "bool", "category"]).columns
    Xe = pd.get_dummies(Xn, columns=cat_cols, drop_first=False, dtype=int)
    return Xe.reindex(columns=final_cols, fill_value=0)

def main():
    p = argparse.ArgumentParser(
        description="Fit/transform pipeline for disk/memory/process + PID/CMD"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    fit = sub.add_parser("fit")
    fit.add_argument("--train-csv",  required=True)
    fit.add_argument("--out-x-csv",  required=True)
    fit.add_argument("--out-y-csv",  required=True)
    fit.add_argument("--preproc-pkl", default="normalize.pkl")

    tr = sub.add_parser("transform")
    tr.add_argument("--in-csv",     required=True)
    tr.add_argument("--out-x-csv",  required=True)
    tr.add_argument("--preproc-pkl", default="normalize.pkl")
    tr.add_argument("--out-y-csv",  default=None)

    args = p.parse_args()

    if args.mode == "fit":
        df = pd.read_csv(args.train_csv)
        y  = df.pop("type")
        X  = df

        scaler, nums, cols, Xe = fit_pipeline(X)

        Xe.to_csv(args.out_x_csv, index=False)
        pd.DataFrame({"type": y}).to_csv(args.out_y_csv, index=False)

        with open(args.preproc_pkl, "wb") as f:
            pickle.dump({
                "scaler":          scaler,
                "numeric_columns": nums,
                "final_columns":   cols
            }, f)

    else:  # transform
        df    = pd.read_csv(args.in_csv)
        has_y = "type" in df.columns
        y     = df.pop("type") if has_y else None
        X     = df

        meta = pickle.load(open(args.preproc_pkl, "rb"))
        Xe   = transform_pipeline(
            X,
            meta["scaler"],
            meta["numeric_columns"],
            meta["final_columns"]
        )

        Xe.to_csv(args.out_x_csv, index=False)

        if has_y and args.out_y_csv:
            pd.DataFrame({"type": y}).to_csv(args.out_y_csv, index=False)

if __name__ == "__main__":
    main()
