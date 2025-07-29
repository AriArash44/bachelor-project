import sys
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from typing import List

def process_ip_csv_to6(input_csv_path: str, temp_dir: str) -> str:
    df = pd.read_csv(input_csv_path)
    def split_into_6(ip: str) -> list[int]:
        if pd.isna(ip):
            return [-1] * 6
        if ":" in ip:
            parts = ip.split(":")
            parts = parts + [""] * (6 - len(parts)) if len(parts) < 6 else parts[:6]
            nums = []
            for p in parts:
                if p == "":
                    nums.append(-1)
                else:
                    try:
                        nums.append(int(p, 16))
                    except ValueError:
                        nums.append(-1)
            return nums
        parts = ip.split(".")
        nums = []
        for p in parts[:4]:
            try:
                nums.append(int(p))
            except ValueError:
                nums.append(-1)
        nums += [-1] * (6 - len(nums))
        return nums
    for col in ["src_ip", "dst_ip"]:
        if col not in df.columns:
            continue
        exploded = df[col].apply(split_into_6)
        for idx in range(6):
            df[f"{col}.part{idx+1}"] = exploded.map(lambda lst: lst[idx])
        df.drop(columns=[col], inplace=True)
    output_path = temp_dir / "0-ip_processed.csv"
    df.to_csv(output_path, index=False)
    return output_path

def one_hot_encode_columns(
    df: pd.DataFrame,
    columns: List[str],
    drop_original: bool = True
) -> pd.DataFrame:
    df_out = df.copy()
    missing = [c for c in columns if c not in df_out.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")
    dummies = pd.get_dummies(df_out[columns], prefix=columns, prefix_sep='.')
    dummies = pd.get_dummies(dummies, drop_first=False).replace({True: 1, False: 0})
    if drop_original:
        df_out.drop(columns=columns, inplace=True)
    return pd.concat([df_out, dummies], axis=1)

def align_with_training_schema(df: pd.DataFrame, pipeline_path: Path) -> pd.DataFrame:
    with open(pipeline_path, "rb") as f:
        scaler = pickle.load(f)
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]

def run_script(script: Path, args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, str(script), *args]
    print(f">> Running {script.name} in {cwd}:\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def dns2numeric(input_csv: str,
                script_path: str,
                pkl_path: str,
                temp_dir: Path) -> Path:
    script = Path(script_path).resolve()
    pkl = Path(pkl_path).resolve()
    inp = Path(input_csv).resolve()
    out = temp_dir / "1-dns2numeric.csv"
    args = [
        "transform",
        "--in-csv", str(inp),
        "--out-csv", str(out),
        "--preproc-pkl", str(pkl),
    ]
    run_script(script, args, cwd=script.parent)
    return out

def uri2numeric(input_csv: str,
                script_path: str,
                pkl_path: str,
                temp_dir: Path) -> Path:
    script = Path(script_path).resolve()
    pkl = Path(pkl_path).resolve()
    inp = Path(input_csv).resolve()
    out = temp_dir / "2-uri2numeric.csv"
    args = [
        "transform",
        "--in-csv", str(inp),
        "--out-csv", str(out),
        "--preproc-pkl", str(pkl),
    ]
    run_script(script, args, cwd=script.parent)
    return out

def uAgent2numeric(input_csv: str,
                script_path: str,
                pkl_path: str,
                temp_dir: Path) -> Path:
    script = Path(script_path).resolve()
    pkl = Path(pkl_path).resolve()
    inp = Path(input_csv).resolve()
    out = temp_dir / "3-uAgent2numeric.csv"
    args = [
        "transform",
        "--in-csv", str(inp),
        "--out-csv", str(out),
        "--preproc-pkl", str(pkl),
    ]
    run_script(script, args, cwd=script.parent)
    return out

def normalize(input_csv: Path,
              script_path: str,
              pkl_path: str,
              temp_dir: Path) -> Path:
    script = Path(script_path).resolve()
    pkl = Path(pkl_path).resolve()
    inp = input_csv.resolve()
    out = temp_dir / "5-X_norm.csv"
    args = [
        "transform",
        "--in-csv", str(inp),
        "--out-x-csv", str(out),
        "--out-y-csv", str(out),
        "--preproc-pkl", str(pkl),
    ]
    run_script(script, args, cwd=script.parent)
    return out

def feature_select(norm_csv: Path,
                   script_path: str,
                   pkl_path: str,
                   temp_dir: Path) -> Path:
    script = Path(script_path).resolve()
    pkl = Path(pkl_path).resolve()
    inp = norm_csv.resolve()
    out = temp_dir / "2-X_preprocessed.csv"
    args = [
        "transform",
        "--in-csv", str(inp),
        "--out-x-csv", str(out),
        "--preproc-pkl", str(pkl),
    ]
    run_script(script, args, cwd=script.parent)
    return out

def slide_windows_with_padding(X: np.ndarray, context: int) -> np.ndarray:
    seq_len = 2 * context + 1
    X_padded = np.pad(X, ((context, context), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(
                   X_padded,
                   window_shape=(seq_len, X.shape[1])
                )
    return windows.squeeze(1)

def predict_direct(selected_csv: Path,
                   model_h5: str,
                   label_map: str,
                   batch_size: int,
                   context: int,
                   temp_dir: Path) -> Path:
    out_y = temp_dir / "3-y_pred.csv"
    df = pd.read_csv(selected_csv)
    X = df.values.astype("float32")
    W = slide_windows_with_padding(X, context=context)
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
        description="ipToNumeric → DNStoNumeric → URItoNumeric → uAgenttoNumeric → oneHotKey → Normalize → Feature-select → Window(pad) → Predict"
    )
    parser.add_argument("input_csv", help="Raw features CSV")
    parser.add_argument("--dns2numeric-script",
                        default="../0-preprocessors/1-numericCategoricals/2-numericDNSQuery/1-dnsToCodebookIndex/dnsToCodebookIndexDriver.py")
    parser.add_argument("--dns2numeric-pkl",
                        default="../0-preprocessors/1-numericCategoricals/2-numericDNSQuery/1-dnsToCodebookIndex/dns_pipeline.pkl")
    parser.add_argument("--uri2numeric-script",
                        default="../0-preprocessors/1-numericCategoricals/3-numericHttpUri/1-uriToCodebookIndex/uriToCodebookIndexDriver.py")
    parser.add_argument("--uri2numeric-pkl",
                        default="../0-preprocessors/1-numericCategoricals/3-numericHttpUri/1-uriToCodebookIndex/uri_pipeline.pkl")
    parser.add_argument("--uAgent2numeric-script",
                        default="../0-preprocessors/1-numericCategoricals/4-numericUAgent/1-uAgentToCodebookIndex/uAgentToCodebookIndexDriver.py")
    parser.add_argument("--uAgent2numeric-pkl",
                        default="../0-preprocessors/1-numericCategoricals/4-numericUAgent/1-uAgentToCodebookIndex/uAgent_pipeline.pkl")    
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
    # parser.add_argument("--model-h5",
    #                 default="../1-AI-model/bigru/bigru_tf.h5")
    # parser.add_argument("--model-h5",
    #                 default="../1-AI-model/mharnn/mharnn_model.h5")
    parser.add_argument("--label-map",
                        default="../1-AI-model/mhabigru/label_map.pkl")
    # parser.add_argument("--label-map",
    #                 default="../1-AI-model/bigru/label_map.pkl")
    # parser.add_argument("--label-map",
    #                 default="../1-AI-model/mharnn/label_map.pkl")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context", type=int, default=10,
                        help="Number of past/future steps on each side")
    args = parser.parse_args(cli_args)
    script_dir = Path(__file__).resolve().parent
    temp_dir = script_dir / "temp_files"
    temp_dir.mkdir(exist_ok=True)
    process_ip_csv_to6(args.input_csv, temp_dir=temp_dir)
    dns2_out = dns2numeric(
        input_csv=temp_dir / "0-ip_processed.csv",
        script_path=args.dns2numeric_script,
        pkl_path=args.dns2numeric_pkl,
        temp_dir=temp_dir,
    )
    uri2_out = uri2numeric(
        input_csv=dns2_out,
        script_path=args.uri2numeric_script,
        pkl_path=args.uri2numeric_pkl,
        temp_dir=temp_dir,
    )
    uAgent2_out = uAgent2numeric(
        input_csv=uri2_out,
        script_path=args.uAgent2numeric_script,
        pkl_path=args.uAgent2numeric_pkl,
        temp_dir=temp_dir,
    )
    df = pd.read_csv(uAgent2_out)
    cols_to_encode = ["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_version", "ssl_cipher",
                      "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer", "http_trans_depth", "http_method", 
                      "http_orig_mime_types", "http_resp_mime_types", "weird_name", "weird_addl", "weird_notice", "http_version"]
    df_encoded = one_hot_encode_columns(df, cols_to_encode)
    df_aligned = align_with_training_schema(df_encoded, args.normalizer_pkl)
    df_aligned.to_csv(temp_dir / "4-oneHot.csv", index=False)
    norm_out = normalize(
        input_csv=temp_dir / "4-oneHot.csv",
        script_path=args.normalizer_script,
        pkl_path=args.normalizer_pkl,
        temp_dir=temp_dir,
    )
    sel_out = feature_select(
        norm_csv=norm_out,
        script_path=args.selector_script,
        pkl_path=args.selector_pkl,
        temp_dir=temp_dir,
    )
    predict_in = sel_out
    predict_direct(
        selected_csv=predict_in,
        model_h5=args.model_h5,
        label_map=args.label_map,
        batch_size=args.batch_size,
        context=args.context,
        temp_dir=temp_dir
    )

if __name__ == "__main__":
    driver_main()
