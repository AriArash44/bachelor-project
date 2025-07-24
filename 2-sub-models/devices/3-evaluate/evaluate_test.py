import sys
from pathlib import Path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
from driver_pkg.driver import driver_main
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score

df = pd.read_csv("../0-preprocessors/1-testTrainSplitter/test_split.csv")
test_Y = df[["type",]].copy()
test_X = df.drop(columns=["type",])

test_X.to_csv("test_X.csv", index=False)

driver_main(["test_X.csv"])

predicted_df = pd.read_csv("../driver_pkg/temp_files/2-y_pred.csv")
predicted_df["predicted"] = predicted_df.idxmax(axis=1)

y_true = test_Y["type"].values
y_pred = predicted_df["predicted"].values

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print(f"✅ Accuracy    : {accuracy:.4f}")
print(f"✅ Precision   : {precision:.4f}")
print(f"✅ F1 score    : {f1:.4f}")