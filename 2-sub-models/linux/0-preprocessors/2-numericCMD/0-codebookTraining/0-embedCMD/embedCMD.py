import pandas as pd
import pickle
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense
from tensorflow.keras.models import Model

def build_vocab(cmd_series, pad_token="<PAD>", unk_token="<UNK>"):
    unique_cmds = cmd_series.dropna().tolist()
    cmd2idx = {cmd: idx for idx, cmd in enumerate(unique_cmds, start=2)}
    cmd2idx[pad_token] = 0
    cmd2idx[unk_token] = 1
    return cmd2idx

def map_cmds(cmd_series, cmd2idx, unk_token="<UNK>"):
    return cmd_series.map(lambda x: cmd2idx.get(x, cmd2idx[unk_token])).astype(int)

def main():
    INPUT_CSV = "../../../1-testTrainSplitter/train_split.csv"
    UNIQUE_EMB_CSV = "cmd_embeddings_unique.csv"
    MODEL_FILE = "cmd_embedding_model_trained.h5"
    VOCAB_FILE = "cmd2idx.pkl"
    EMBEDDING_DIM = 64
    EPOCHS = 10
    BATCH_SIZE = 32
    df = pd.read_csv(INPUT_CSV)
    df['CMD'] = (
        df['CMD']
          .astype(str)
          .str.strip()
          .str.replace(r'\s+', ' ', regex=True)
          .str.lower()
    )
    unique_cmds = df['CMD'].drop_duplicates().reset_index(drop=True)
    cmd2idx = build_vocab(unique_cmds)
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump(cmd2idx, f)
    ids = map_cmds(unique_cmds, cmd2idx).to_numpy().reshape(-1, 1)
    labels = ids.copy()
    vocab_size = len(cmd2idx)
    inp = Input(shape=(1,), dtype='int32')
    emb = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(inp)
    flat = Flatten()(emb)
    out = Dense(vocab_size, activation='softmax')(flat)
    clf = Model(inp, out)
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf.fit(ids, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    emb_weights = clf.get_layer(index=1).get_weights()[0]
    unique_ids = [cmd2idx[cmd] for cmd in unique_cmds]
    emb_np_unique = emb_weights[unique_ids]
    emb_cols = [f'cmd_emb_{i}' for i in range(EMBEDDING_DIM)]
    df_embs = pd.DataFrame(emb_np_unique, columns=emb_cols)
    df_embs.to_csv(UNIQUE_EMB_CSV, index=False)
    clf.save(MODEL_FILE)
    print("✅ Unique embeddings saved to", UNIQUE_EMB_CSV)
    print("✅ Embedding model saved to", MODEL_FILE)
    print("✅ cmd2idx mapping saved to", VOCAB_FILE)

if __name__ == "__main__":
    main()
