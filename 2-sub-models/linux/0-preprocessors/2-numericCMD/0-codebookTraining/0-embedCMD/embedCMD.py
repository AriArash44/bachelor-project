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
    df_unique = df[['CMD']].drop_duplicates().reset_index(drop=True)
    cmd2idx = build_vocab(df_unique['CMD'])
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump(cmd2idx, f)
    df_unique['cmd_id'] = map_cmds(df_unique['CMD'], cmd2idx)
    vocab_size = len(cmd2idx)
    inp = Input(shape=(1,), dtype='int32', name='cmd_id_input')
    emb = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True, name='cmd_embedding')(inp)
    flat = Flatten()(emb)
    out = Dense(vocab_size, activation='softmax')(flat)
    clf = Model(inp, out)
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    ids = df_unique['cmd_id'].to_numpy().reshape(-1, 1)
    labels = ids.copy()
    clf.fit(ids, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    emb_weights = clf.get_layer('cmd_embedding').get_weights()[0]
    unique_ids = [cmd2idx[cmd] for cmd in df_unique['CMD']]
    emb_np_unique = emb_weights[unique_ids]
    emb_cols = [f'cmd_emb_{i}' for i in range(EMBEDDING_DIM)]
    df_map = pd.DataFrame({col: emb_np_unique[:, i] for i, col in enumerate(emb_cols)})
    df_map.to_csv(UNIQUE_EMB_CSV, index=False)
    clf.save(MODEL_FILE)
    print("✅ Unique embeddings saved to", UNIQUE_EMB_CSV)
    print("✅ Embedding model saved to", MODEL_FILE)
    print("✅ cmd2idx mapping saved to", VOCAB_FILE)

if __name__ == "__main__":
    main()
