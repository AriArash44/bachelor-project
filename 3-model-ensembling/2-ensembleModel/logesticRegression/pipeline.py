import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import joblib
import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))
from customTransformer.featureScaler import FeatureScaler

def main(args):
    df_dev = pd.read_csv(args.devices_csv)
    df_net = pd.read_csv(args.network_csv)
    df_lin = pd.read_csv(args.linux_csv)
    df_types = pd.read_csv(args.type_csv)
    df_dev.columns = [f"dev_{c}" for c in df_dev.columns]
    df_net.columns = [f"net_{c}" for c in df_net.columns]
    df_lin.columns = [f"lin_{c}" for c in df_lin.columns]
    X = pd.concat([df_dev, df_net, df_lin], axis=1)
    y = df_types[args.label_column]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    dev_cols = df_dev.columns.tolist()
    net_cols = df_net.columns.tolist()
    lin_cols = df_lin.columns.tolist()
    preprocessor = ColumnTransformer([
        ("dev", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("weight", FeatureScaler(args.weight_dev)),
            ("scale", StandardScaler())
        ]), dev_cols),
        ("net", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("weight", FeatureScaler(args.weight_net)),
            ("scale", StandardScaler())
        ]), net_cols),
        ("lin", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("weight", FeatureScaler(args.weight_lin)),
            ("scale", StandardScaler())
        ]), lin_cols),
    ])
    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=args.max_iter,
            random_state=args.random_state
        ))
    ])
    pipeline.fit(X, y_enc)
    joblib.dump(pipeline, args.output_model)
    joblib.dump(le, args.output_model.replace(".pkl", "_label_encoder.pkl"))
    print(f"✅ model saved to {args.output_model}")
    print(f"✅ label encoder saved to {args.output_model.replace('.pkl', '_label_encoder.pkl')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train weighted LogisticRegression and dump the pipeline"
    )
    parser.add_argument("--devices_csv", required=True, help="path to device probabilities CSV")
    parser.add_argument("--network_csv", required=True, help="path to network probabilities CSV")
    parser.add_argument("--linux_csv", required=True, help="path to linux probabilities CSV")
    parser.add_argument("--type_csv", required=True, help="path to CSV with true labels")
    parser.add_argument("--output_model", required=True, help="where to save the trained pipeline (.pkl)")
    parser.add_argument("--label_column", default="type", help="column name of target in type_csv")
    parser.add_argument("--weight_dev", type=float, default=0.9, help="weight for device features")
    parser.add_argument("--weight_net", type=float, default=0.95, help="weight for network features")
    parser.add_argument("--weight_lin", type=float, default=0.7, help="weight for linux features")
    parser.add_argument("--max_iter", type=int, default=1000, help="max iterations for LogisticRegression")
    parser.add_argument("--random_state",type=int, default=42, help="random seed")
    args = parser.parse_args()
    main(args)