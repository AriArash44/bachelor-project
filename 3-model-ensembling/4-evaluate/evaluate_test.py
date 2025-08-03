import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
from driver_pkg.driver import driver_main

TEST_DATA_CSV = "../0-testTrainMerge/margedTest.csv"
TMP_INPUT_CSV = "test_X.csv"
TMP_Y_CSV = "test_Y.csv"
LABEL_NORMAL  = "normal"

def evaluate():
    df = pd.read_csv(TEST_DATA_CSV)
    y_true_df = df[["type"]].copy()
    X_df = df.drop(columns=["type"])
    X_df.to_csv(TMP_INPUT_CSV, index=False)
    y_true_df.to_csv(TMP_Y_CSV, index=False)
    driver_main([
        "--input-csv", TMP_INPUT_CSV,
        "--model-pkl",  "../2-ensembleModel/logesticRegression/model.pkl",
        "--label-map",  "../2-ensembleModel/logesticRegression/model_label_encoder.pkl",
        # "--model-pkl",  "../2-ensembleModel/xgBoost/model.pkl",
        # "--label-map",  "../2-ensembleModel/xgBoost/model_label_encoder.pkl",
    ])
    pred_df = pd.read_csv("../driver_pkg/tmp_files/output.csv")
    y_pred = pred_df["predicted"].values
    y_true = y_true_df["type"].values
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"✅ Accuracy : {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ F1 Score : {f1:.4f}")
    bin_true = [0 if lbl == LABEL_NORMAL else 1 for lbl in y_true]
    bin_pred = [0 if lbl == LABEL_NORMAL else 1 for lbl in y_pred]
    bin_acc = accuracy_score(bin_true, bin_pred)
    bin_prec = precision_score(bin_true, bin_pred, average="binary", zero_division=0)
    bin_f1 = f1_score(bin_true, bin_pred, average="binary", zero_division=0)
    print("\n🔄 Binary Evaluation (Normal vs Attack)")
    print(f"⚡ Accuracy : {bin_acc:.4f}")
    print(f"⚡ Precision: {bin_prec:.4f}")
    print(f"⚡ F1 Score : {bin_f1:.4f}")

if __name__ == "__main__":
    evaluate()
