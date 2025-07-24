import sys
from pathlib import Path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))
from driver_pkg.driver import driver_main
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np

# 1) آماده‌سازی داده‌های تست
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

df = pd.read_csv("../0-preprocessors/1-testTrainSplitter/test_split.csv")
test_Y = df["type"].copy()    # آرایهٔ اصلی با طول ~30k
test_X = df.drop(columns=["type"])
test_X.to_csv("test_X.csv", index=False)

# 2) فراخوانی درایور و دریافت پیش‌بینی پنجره‌ای
driver_main(["test_X.csv"])
predicted_df = pd.read_csv("../driver_pkg/temp_files/2-y_pred.csv")
# اینجا فرض بر این است که ستون "predicted" حاوی برچسب argmax برای هر پنجره است

# 3) گروه‌بندی ۲۰تایی برچسب‌ها
group_size = 20

# اگر تعداد total_y یا total_pred مضرب group_size نیست،
# ابتدا آن را به یکسان برش بزنید تا تعداد نمونه‌ها در هر دو برابر شود.
min_len = min(len(test_Y), len(predicted_df))
test_Y = test_Y.iloc[:min_len].reset_index(drop=True)
predicted = predicted_df["predicted"].iloc[:min_len].reset_index(drop=True)

# شاخص گروه برای هر نمونه
groups = np.arange(min_len) // group_size

# y_true گروه‌بندی‌شده با اکثریت (mode) هر بلوک
y_true_grouped = (
    test_Y
      .groupby(groups)
      .agg(lambda x: x.value_counts().idxmax())
      .values
)

# y_pred گروه‌بندی‌شده با اکثریت هر بلوک
y_pred_grouped = (
    predicted
      .groupby(groups)
      .agg(lambda x: x.value_counts().idxmax())
      .values
)

# 4) اعمال متریک‌ها روی آرایه‌های هم‌طول
accuracy  = accuracy_score(y_true_grouped, y_pred_grouped)
precision = precision_score(y_true_grouped, y_pred_grouped,
                            average="macro", zero_division=0)
f1        = f1_score(y_true_grouped, y_pred_grouped,
                     average="macro", zero_division=0)

print(f"✅ Accuracy  : {accuracy:.4f}")
print(f"✅ Precision : {precision:.4f}")
print(f"✅ F1 score  : {f1:.4f}")
