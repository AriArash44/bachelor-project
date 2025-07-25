import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
sys.path.append("../0-codebookTraining/1-VAE")
from vaeEmbedVectors import KLDivergenceLayer, sampling

VOCAB_PKL = "../0-codebookTraining/0-embedCMD/cmd2idx.pkl"
EMB_MODEL_H5 = "../0-codebookTraining/0-embedCMD/cmd_embedding_model_trained.h5"
ENCODER_MU_MODEL = "../0-codebookTraining/1-VAE/encoder_z.h5"
KMEANS_CSV = "../0-codebookTraining/3-codebook_ordering/kmeans_centers_32_ordered.csv"
UNK_TOKEN = "<UNK>"

class CmdEncodingPipeline:
    def __init__(self, cmd2idx: dict):
        self.cmd2idx = cmd2idx
        self._emb_model = load_model(EMB_MODEL_H5)
        self._enc_mu_model = load_model(ENCODER_MU_MODEL, custom_objects={
            "KLDivergenceLayer": KLDivergenceLayer,
            "sampling": sampling
        })
        self.kmeans_centers = pd.read_csv(KMEANS_CSV).values.astype(np.float32)

    def _safe_lookup(self, cmd: str) -> int:
        return self.cmd2idx.get(cmd, self.cmd2idx.get(UNK_TOKEN, 1))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ids = (
            df["CMD"]
              .map(self._safe_lookup)
              .astype(int)
              .values
              .reshape(-1, 1)
        )
        emb_layer = self._emb_model.get_layer("embedding")
        emb_matrix = emb_layer.get_weights()[0]
        cmd_ids = ids.squeeze()
        emb = emb_matrix[cmd_ids]
        mus = self._enc_mu_model.predict(emb, verbose=0)
        dists = np.sum(
            (mus[:, None, :] - self.kmeans_centers[None, :, :])**2,
            axis=2
        )
        nearest_idx = np.argmin(dists, axis=1)
        out = df.copy()
        out["CMD"] = nearest_idx
        return out

def main():
    p = argparse.ArgumentParser(description="Load vocab from pickle + Fit/Transform CMD→codebook pipeline")
    subs = p.add_subparsers(dest="mode", required=True)
    fit = subs.add_parser("fit", help="Load cmd2idx.pkl, transform train, save pipeline")
    fit.add_argument("--train-csv", required=True, help="input dataset file")
    fit.add_argument("--out-csv", required=True, help="output dataset file")
    fit.add_argument("--preproc-pkl", default="cmd_pipeline.pkl", help="path of pickle pipeline")
    tr = subs.add_parser("transform", help="Load pipeline.pkl and transform CSV")
    tr.add_argument("--in-csv", required=True, help="input dataset file")
    tr.add_argument("--out-csv", required=True, help="output dataset file")
    tr.add_argument("--preproc-pkl", default="cmd_pipeline.pkl", help="path of pickle pipeline")
    args = p.parse_args()

    if args.mode == "fit":
        with open(VOCAB_PKL, "rb") as f:
            cmd2idx = pickle.load(f)
        pipeline = CmdEncodingPipeline(cmd2idx)
        df = pd.read_csv(args.train_csv)
        df_out = pipeline.transform(df)
        df_out.to_csv(args.out_csv, index=False)
        with open(args.preproc_pkl, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"✅ Fit complete. Transformed → {args.out_csv}")
        print(f"✅ Pipeline saved → {args.preproc_pkl}")

    else:
        with open(args.preproc_pkl, "rb") as f:
            pipeline = pickle.load(f)
        df = pd.read_csv(args.in_csv)
        df_out = pipeline.transform(df)
        df_out.to_csv(args.out_csv, index=False)
        print(f"✅ Transform complete. Output → {args.out_csv}")

if __name__ == "__main__":
    main()
