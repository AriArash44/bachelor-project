import sys
from pathlib import Path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from driver_pkg.driver import driver_main
import pandas as pd

df = pd.read_csv("../0-preprocessors/1-testTrainSplitter/test_split.csv")
test_Y = df[["type",]].copy()
test_X = df.drop(columns=["type",])

test_X.to_csv("test_X.csv", index=False)
test_Y.to_csv("test_Y.csv", index=False)

predict_Y = driver_main(["test_X.csv"])