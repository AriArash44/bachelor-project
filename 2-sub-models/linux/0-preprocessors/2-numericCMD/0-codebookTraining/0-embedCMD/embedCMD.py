import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.models import Model

def build_vocab(cmd_series, pad_token="<PAD>"):
    unique_cmds = cmd_series.dropna().unique().tolist()
    cmd2idx = {cmd: idx for idx, cmd in enumerate(unique_cmds, start=1)}
    cmd2idx[pad_token] = 0
    return cmd2idx

def build_embedding_model(vocab_size: int, embedding_dim: int = 64) -> Model:
    input_ids = Input(shape=(1,), dtype="int32", name="cmd_id_input")
    emb_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="cmd_embedding"
    )(input_ids)
    model = Model(inputs=input_ids, outputs=emb_layer, name="cmd_embedding_model")
    return model

def main():
    INPUT_CSV = "../../../1-testTrainSplitter/train_split.csv"
    OUTPUT_CSV = "cmd_embeddings_only.csv"
    MODEL_FILE = "cmd_embedding_model.h5"
    EMBEDDING_DIM = 64
    df = pd.read_csv(INPUT_CSV)
    cmd2idx = build_vocab(df["CMD"])
    df["cmd_id"] = df["CMD"].map(cmd2idx).fillna(0).astype(int)
    vocab_size = len(cmd2idx)
    cmd_ids = df["cmd_id"].values
    emb_model = build_embedding_model(vocab_size, EMBEDDING_DIM)
    emb_np = emb_model.predict(cmd_ids, verbose=0).squeeze()
    emb_cols = [f"cmd_emb_{i}" for i in range(EMBEDDING_DIM)]
    df_emb = pd.DataFrame(emb_np, columns=emb_cols)
    df_emb.to_csv(OUTPUT_CSV, index=False)
    emb_model.save(MODEL_FILE)
    print(f"✅ Saved embeddings to {OUTPUT_CSV}")
    print(f"✅ Saved embedding model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
