import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1.0):
        self.weight = weight

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.weight

def main(args):
    df_dev = pd.read_csv(args.devices_csv)
    df_net = pd.read_csv(args.network_csv)
    df_lin = pd.read_csv(args.linux_csv)
    df_types = pd.read_csv(args.type_csv)
    df_dev.columns = [f'dev_{c}' for c in df_dev.columns]
    df_net.columns = [f'net_{c}' for c in df_net.columns]
    df_lin.columns = [f'lin_{c}' for c in df_lin.columns]
    X = pd.concat([df_dev, df_net, df_lin], axis=1)
    y = df_types[args.label_column]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    dev_cols = df_dev.columns.tolist()
    net_cols = df_net.columns.tolist()
    lin_cols = df_lin.columns.tolist()
    preprocessor = ColumnTransformer([
        ('dev', Pipeline([
            ('weigh', FeatureScaler(args.weight_dev)),
            ('scale', StandardScaler())
        ]), dev_cols),
        ('net', Pipeline([
            ('weigh', FeatureScaler(args.weight_net)),
            ('scale', StandardScaler())
        ]), net_cols),
        ('lin', Pipeline([
            ('weigh', FeatureScaler(args.weight_lin)),
            ('scale', StandardScaler())
        ]), lin_cols),
    ])
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', XGBClassifier(
            objective='multi:softprob',
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth
        )),
    ])
    pipeline.fit(X, y_enc)
    joblib.dump(pipeline, args.output_model)
    joblib.dump(le, args.output_model.replace('.pkl', '_label_encoder.pkl'))
    print(f"✅ model saved to {args.output_model}")
    print(f"✅ label encoder saved to {args.output_model.replace('.pkl', '_label_encoder.pkl')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost classifier on device/network/linux probabilities"
    )
    parser.add_argument('--devices_csv', required=True)
    parser.add_argument('--network_csv', required=True)
    parser.add_argument('--linux_csv', required=True)
    parser.add_argument('--type_csv', required=True)
    parser.add_argument('--output_model', required=True, help="Path to save trained pipeline (e.g. model.pkl)")
    parser.add_argument('--label_column', default='type')
    parser.add_argument('--weight_dev', type=float, default=0.9)
    parser.add_argument('--weight_net', type=float, default=0.95)
    parser.add_argument('--weight_lin', type=float, default=0.7)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=6)
    args = parser.parse_args()
    main(args)
