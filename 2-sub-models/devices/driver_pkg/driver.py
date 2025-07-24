import subprocess
import argparse
from pathlib import Path

def normalize(input_csv: str,
              normalizer_script: str,
              normalizer_pkl: str,
              temp_dir: Path) -> Path:
    out_x = temp_dir / "0-X_norm.csv"
    cmd = [
        "py", normalizer_script,
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
        "py", selector_script,
        "transform",
        "--in-csv", str(norm_csv),
        "--out-x-csv", str(out_x),
        "--preproc-pkl", selector_pkl,
    ]
    print(">> Feature selecting:\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_x

def predict(selected_csv: Path,
            model_script: str,
            model_pt: str,
            label_map: str,
            temp_dir: Path) -> Path:
    out_y = temp_dir / "2-y_pred.csv"
    cmd = [
        "py", model_script,
        "--in-csv", str(selected_csv),
        "--out-y-csv", str(out_y),
        "--model-pt", model_pt,
        "--label-map", label_map,
    ]
    print(">> Predicting Y:\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_y

def driver_main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="Normalize → Feature-select → Predict"
    )
    parser.add_argument("input_csv", help="Raw features CSV")

    parser.add_argument(
        "--normalizer-script",
        default="../0-preprocessors/2-normalizer/normalizer.py",
        help="Path to normalizer.py",
    )
    parser.add_argument(
        "--normalizer-pkl",
        default="../0-preprocessors/2-normalizer/normalize.pkl",
        help="Pickle for normalizer",
    )
    parser.add_argument(
        "--selector-script",
        default="../0-preprocessors/3-featureSelector/featureSelector.py",
        help="Path to featureSelector.py",
    )
    parser.add_argument(
        "--selector-pkl",
        default="../0-preprocessors/3-featureSelector/feature_selection.pkl",
        help="Pickle for feature-selection pipeline",
    )
    parser.add_argument(
        "--model-script",
        default="../1-AI-model/mhabigru/model.py",
        help="Path to your CLI entrypoint for predict",
    )
    parser.add_argument(
        "--model-pt",
        default="../1-AI-model/mhabigru/mhabigru_model.pt",
        help="Path to your PyTorch .pt checkpoint",
    )
    parser.add_argument(
        "--label-map",
        default="../1-AI-model/mhabigru/label_map.pkl",
        help="Pickle for mapping label IDs → human labels",
    )
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
    selected_path = feature_select(
        norm_csv=norm_path,
        selector_script=args.selector_script,
        selector_pkl=args.selector_pkl,
        temp_dir=temp_dir,
    )
    y_path = predict(
        selected_csv=selected_path,
        model_script=args.model_script,
        model_pt=args.model_pt,
        label_map=args.label_map,
        temp_dir=temp_dir,
    )
    print(f">> Finished. Y predictions saved to: {y_path}")

if __name__ == "__main__":
    driver_main()
