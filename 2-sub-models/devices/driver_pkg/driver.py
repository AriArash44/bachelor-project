import subprocess
import argparse
from pathlib import Path

def normalize(input_csv: str,
              normalizer_script: str,
              normalizer_pkl: str,
              temp_dir: Path) -> Path:
    out_x = temp_dir / "0-X_norm.csv"
    cmd = [
        "py",
        normalizer_script,
        "transform",
        "--in-csv",      str(input_csv),
        "--out-x-csv",   str(out_x),
        "--preproc-pkl", normalizer_pkl
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
        "py",
        selector_script,
        "transform",
        "--in-csv",      str(norm_csv),
        "--out-x-csv",   str(out_x),
        "--preproc-pkl", selector_pkl
    ]
    print(">> Feature selecting:\n   " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_x

def driver_main(cli_args=None):
    parser = argparse.ArgumentParser(
        description="1) Normalize → temp_files/0-X_norm.csv\n"
                    "2) Feature-select → temp_files/1-X_preprocessed.csv"
    )
    parser.add_argument("input_csv", help="Raw features CSV")
    parser.add_argument(
        "--normalizer-script",
        default="../0-preprocessors/2-normalizer/normalizer.py",
        help="Path to normalizer.py"
    )
    parser.add_argument(
        "--normalizer-pkl",
        default="../0-preprocessors/2-normalizer/normalize.pkl",
        help="Pickle for normalizer"
    )
    parser.add_argument(
        "--selector-script",
        default="../0-preprocessors/3-featureSelector/featureSelector.py",
        help="Path to featureSelector.py"
    )
    parser.add_argument(
        "--selector-pkl",
        default="../0-preprocessors/3-featureSelector/feature_selection.pkl",
        help="Pickle for feature-selection pipeline"
    )
    args = parser.parse_args(cli_args)
    script_dir = Path(__file__).resolve().parent
    temp_dir = script_dir / "temp_files"
    temp_dir.mkdir(exist_ok=True)
    norm_path = normalize(
        input_csv=args.input_csv,
        normalizer_script=args.normalizer_script,
        normalizer_pkl=args.normalizer_pkl,
        temp_dir=temp_dir
    )
    final_path = feature_select(
        norm_csv=norm_path,
        selector_script=args.selector_script,
        selector_pkl=args.selector_pkl,
        temp_dir=temp_dir
    )
    
if __name__ == "__main__":
    driver_main()
