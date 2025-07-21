import subprocess
import argparse
import os
from pathlib import Path

def transform_to_temp(input_csv: str, preproc_pkl: str = "../0-preprocessors/2-normalizer/normalize.pkl") -> str:
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "temp_files"
    out_dir.mkdir(exist_ok=True)
    out_x = os.path.join(out_dir, "X_norm.csv")
    cmd = [
        "py", "../0-preprocessors/2-normalizer/normalizer.py", "transform",
        "--in-csv", input_csv,
        "--out-x-csv", out_x,
        "--preproc-pkl", preproc_pkl
    ]
    subprocess.run(cmd, check=True)
    return out_x

def driver_main(cli_args=None):
    parser = argparse.ArgumentParser(description="Normalize a CSV into temp X_norm.csv")
    parser.add_argument("input_csv", help = "Raw features CSV")
    parsed = parser.parse_args(cli_args)
    transform_to_temp(parsed.input_csv)