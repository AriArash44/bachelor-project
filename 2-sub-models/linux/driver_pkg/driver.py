import sys
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

def run_script(script: Path, args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, str(script), *args]
    print(f">> Running {script.name} in {cwd}:\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)

def cmd2numeric(input_csv: str,
                script_path: str,
                pkl_path: str,
                temp_dir: Path) -> Path:
    script = Path(script_path).resolve()
    pkl = Path(pkl_path).resolve()
    inp = Path(input_csv).resolve()
    out = temp_dir / "0-cmd2numeric.csv"
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
    out = temp_dir / "1-X_norm.csv"
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
        description="cmd2numeric → Normalize → Feature-select → Window(pad) → Predict"
    )
    parser.add_argument("input_csv", help="Raw features CSV")
    parser.add_argument("--cmd2numeric-script",
                        default="../0-preprocessors/2-numericCMD/1-cmdToCodebookIndex/cmdToCodebookIndexDriver.py")
    parser.add_argument("--cmd2numeric-pkl",
                        default="../0-preprocessors/2-numericCMD/1-cmdToCodebookIndex/cmd_pipeline.pkl")
    parser.add_argument("--normalizer-script",
                        default="../0-preprocessors/3-normalizer/normalizer.py")
    parser.add_argument("--normalizer-pkl",
                        default="../0-preprocessors/3-normalizer/normalize.pkl")
    parser.add_argument("--selector-script",
                        default="../0-preprocessors/4-featureSelector/featureSelector.py")
    parser.add_argument("--selector-pkl",
                        default="../0-preprocessors/4-featureSelector/feature_selection.pkl")
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
    cmd2_out = cmd2numeric(
        input_csv=args.input_csv,
        script_path=args.cmd2numeric_script,
        pkl_path=args.cmd2numeric_pkl,
        temp_dir=temp_dir,
    )
    norm_out = normalize(
        input_csv=cmd2_out,
        script_path=args.normalizer_script,
        pkl_path=args.normalizer_pkl,
        temp_dir=temp_dir,
    )
    # sel_out = feature_select(
    #     norm_csv=norm_out,
    #     script_path=args.selector_script,
    #     pkl_path=args.selector_pkl,
    #     temp_dir=temp_dir,
    # )
    # predict_in = sel_out
    predict_in = norm_out
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
