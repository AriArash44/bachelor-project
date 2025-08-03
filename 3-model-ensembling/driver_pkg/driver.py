import sys
import argparse
import joblib
import pandas as pd
from pathlib import Path
from multiprocessing import freeze_support
proj_root = Path(__file__).parent.parent / "1-datasetSplit"
sys.path.insert(0, str(proj_root))
from splitCaller import split, calculate_possibilities
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from customTransformer.featureScaler import FeatureScaler

def driver_main(cli_args=None):
    parser = argparse.ArgumentParser(description="Split → calculate_possibilities → predict")
    parser.add_argument("--input-csv", required=True, help="Path to raw features CSV")
    parser.add_argument("--model-pkl", required=True, help="Path to ensemble model pickle file")
    parser.add_argument("--label-map", required=True, help="Path to label encoder pickle file")
    parser.add_argument("--submodels-dir", type=Path, default=proj_root.parent.parent / "2-sub-models", help="Root directory of sub-model drivers")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "tmp_files",  help="Directory for intermediate & final CSVs")
    args = parser.parse_args(cli_args)
    raw_df = pd.read_csv(args.input_csv)
    dev_df, lin_df, net_df = split(raw_df)
    calculate_possibilities(dev_df, lin_df, net_df, submodels_dir=args.submodels_dir,  output_dir=args.output_dir)
    parts = [
        pd.read_csv(args.output_dir / f"{name}.csv")
        for name in ("devices", "linux", "network")
    ]
    parts[0].columns = [f"dev_{c}" for c in parts[0].columns]
    parts[1].columns = [f"lin_{c}" for c in parts[1].columns]
    parts[2].columns = [f"net_{c}" for c in parts[2].columns]
    X_new = pd.concat(parts, axis=1)
    pipeline = joblib.load(args.model_pkl)
    label_enc = joblib.load(args.label_map)
    y_idx = pipeline.predict(X_new)
    y_proba = pipeline.predict_proba(X_new)
    y_lab = label_enc.inverse_transform(y_idx)
    out_df = X_new.copy()
    out_df["predicted"] = y_lab
    for i, cls in enumerate(label_enc.classes_):
        out_df[f"proba_{cls}"] = y_proba[:, i]
    out_path = args.output_dir / "output.csv"
    out_df.to_csv(out_path, index=False)
    print(f"✅ Final predictions saved to {out_path}")

if __name__ == "__main__":
    freeze_support()
    driver_main()
