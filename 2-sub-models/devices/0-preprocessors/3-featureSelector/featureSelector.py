import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = []
        for col in upper.columns:
            for row in upper.index:
                val = upper.loc[row, col]
                if val > self.threshold:
                    high_corr.append((row, col, val))
        print(f"\nHighly Correlated Pairs (thr > {self.threshold}):")
        for a, b, v in high_corr:
            print(f"  {a} ↔ {b}: {v:.4f}")
        self.columns_to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]
        print(f"\nColumns to drop ({len(self.columns_to_drop_)}):")
        print(" ", self.columns_to_drop_)
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

def do_fit(args):
    X = pd.read_csv(args.train_x_csv)
    Y = pd.read_csv(args.train_y_csv)
    pipeline = Pipeline([
        ("dropper", CorrelationDropper(threshold=args.threshold)),
        ("pca", PCA(n_components=args.pca_variance, svd_solver="full")),
        ("scaler", MinMaxScaler()),
    ])
    X_t = pipeline.fit_transform(X)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(pipeline, f)
    n_pcs = pipeline.named_steps["pca"].n_components_
    cols = [f"PC{i+1}" for i in range(n_pcs)]
    df_out = pd.DataFrame(X_t, columns=cols)
    df_out = pd.concat([df_out, Y.reset_index(drop=True)], axis=1)
    df_out.to_csv(args.out_csv, index=False)
    evr = pipeline.named_steps["pca"].explained_variance_ratio_
    cum = np.cumsum(evr)
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1,len(cum)+1), cum, marker="o")
    plt.xlabel("Components")
    plt.ylabel("Cumulative Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.plot_file)
    plt.close()

def do_transform(args):
    with open(args.preproc_pkl, "rb") as f:
        pipeline = pickle.load(f)
    X = pd.read_csv(args.in_csv)
    X_t = pipeline.transform(X)
    n_pcs = pipeline.named_steps["pca"].n_components_
    cols = [f"PC{i+1}" for i in range(n_pcs)]
    df_out = pd.DataFrame(X_t, columns=cols)
    df_out.to_csv(args.out_x_csv, index=False)

p = argparse.ArgumentParser(
    description="fSelection‐Pipeline: fit or transform"
)
sub = p.add_subparsers(dest="mode", required=True)
fit = sub.add_parser("fit")
fit.add_argument("--train-x-csv", required=True)
fit.add_argument("--train-y-csv", required=True)
fit.add_argument("--out-csv", default="preprocessed.csv")
fit.add_argument("--out-pkl", default="feature_selection.pkl")
fit.add_argument("--plot-file", default="pca_explained_variance.png")
fit.add_argument("--threshold", type=float, default=0.95)
fit.add_argument("--pca-variance", type=float, default=0.90)
tr = sub.add_parser("transform")
tr.add_argument("--in-csv", required=True)
tr.add_argument("--out-x-csv", required=True)
tr.add_argument("--preproc-pkl", required=True)
args = p.parse_args()
if args.mode == "fit":
    do_fit(args)
else:
    do_transform(args)