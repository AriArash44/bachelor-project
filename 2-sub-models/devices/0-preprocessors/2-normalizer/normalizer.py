import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

time_columns = [
    "fridge.datetime", "garage_door.datetime", "gps_tracker.datetime",
    "modbus.datetime", "motion_light.datetime", "thermostat.datetime",
    "weather.datetime"
]

devices = [
    "fridge", "garage_door", "gps_tracker",
    "modbus", "motion_light", "thermostat", "weather"
]

baseline_date = "2019-03-31 00:00:00"

cat_mappings = {
    "fridge.temp_condition": {"high": 1, "low": 0},
    "garage_door.door_state": {"open": 1, "closed": 0},
    "garage_door.sphone_signal": {"true": 1, "false": 0},
    "motion_light.light_status": {"on": 1, "off": 0}
}

def backfill_time(df):
    df_time = df.apply(lambda c: pd.to_datetime(c, errors="coerce"))
    stacked = df_time.stack(dropna=False).bfill()
    filled = stacked.fillna(df_time.max().max())
    return filled.unstack()

def normalize(df):
    for dev in devices:
        cols = [c for c in df.columns if c.startswith(dev + ".")]
        df[f"{dev}.is_off"] = df[cols].isnull().all(axis=1).astype(int)
    for col, m in cat_mappings.items():
        df[col] = df[col].astype(str).str.strip().map(m).fillna(0).astype(int)
    num_cols = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns
        if any(c.startswith(dev + ".") for dev in devices)
    ]
    for c in num_cols:
        if c.endswith(".is_off"):
            continue
        dev = c.split(".")[0]
        off = df[f"{dev}.is_off"] == 1
        on = ~off
        mn = df.loc[on, c].min()
        mv = df.loc[on, c].mean() or 0
        df.loc[on, c] = df.loc[on, c].fillna(mv)
        df.loc[off, c] = df.loc[off, c].fillna(mn - 1)
    raw = df[time_columns].copy()
    filled = backfill_time(raw)
    for c in time_columns:
        dev = c.split(".")[0]
        off = df[f"{dev}.is_off"] == 1
        df.loc[off,   c] = baseline_date
        df.loc[~off,  c] = filled[c]
    df[time_columns] = df[time_columns].apply(
        pd.to_datetime, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    base_ts = pd.Timestamp(baseline_date)
    for c in time_columns:
        df[c] = df[c].apply(
            lambda x: (x - base_ts).total_seconds() if pd.notnull(x) else np.nan
        )
    return df

def fit_pipeline(X):
    Xn = normalize(X.copy())
    nums = [
        c for c in Xn.select_dtypes(include=["int64", "float64"]).columns
        if any(c.startswith(dev + ".") for dev in devices)
    ]
    scaler = MinMaxScaler()
    Xn[nums] = scaler.fit_transform(Xn[nums])
    Xe = pd.get_dummies(Xn, drop_first=False).replace({True: 1, False: 0})
    return scaler, nums, Xe.columns.tolist(), Xe

def transform_pipeline(X, scaler, nums, cols):
    Xn = normalize(X.copy())
    Xn[nums] = scaler.transform(Xn[nums])
    Xe = pd.get_dummies(Xn, drop_first=False).replace({True: 1, False: 0})
    return Xe.reindex(columns=cols, fill_value=0)

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="mode", required=True)
fit = sub.add_parser("fit")
fit.add_argument("--train-csv", required=True)
fit.add_argument("--out-x-csv", required=True)
fit.add_argument("--out-y-csv", required=True)
fit.add_argument("--preproc-pkl", default="normalize.pkl")
tr = sub.add_parser("transform")
tr.add_argument("--in-csv", required=True)
tr.add_argument("--out-x-csv", required=True)
tr.add_argument("--preproc-pkl", default="normalize.pkl")
args = p.parse_args()
if args.mode == "fit":
    df = pd.read_csv(args.train_csv)
    y = df.pop("type")
    X = df
    scaler, nums, cols, Xe = fit_pipeline(X)
    Xe.to_csv(args.out_x_csv, index=False)
    pd.DataFrame({"type": y}).to_csv(args.out_y_csv, index=False)
    with open(args.preproc_pkl, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "numeric_columns": nums,
            "final_columns":   cols
        }, f)
else:
    df = pd.read_csv(args.in_csv)
    has_y = "type" in df.columns
    y = df.pop("type") if has_y else None
    X = df
    meta = pickle.load(open(args.preproc_pkl, "rb"))
    Xe = transform_pipeline(
        X,
        meta["scaler"],
        meta["numeric_columns"],
        meta["final_columns"]
    )
    Xe.to_csv(args.out_x_csv, index=False)
    if has_y and args.out_y_csv:
        pd.DataFrame({"type": y}).to_csv(args.out_y_csv, index=False)