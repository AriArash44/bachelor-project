import pandas as pd
import pickle
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense
from tensorflow.keras.models import Model

def build_vocab(uAgent_series, pad_token="<PAD>", unk_token="<UNK>"):
    unique_uAgents = uAgent_series.dropna().tolist()
    uAgent2idx = {uAgent: idx for idx, uAgent in enumerate(unique_uAgents, start=2)}
    uAgent2idx[pad_token] = 0
    uAgent2idx[unk_token] = 1
    return uAgent2idx

def map_uAgents(uAgent_series, uAgent2idx, unk_token="<UNK>"):
    return uAgent_series.map(lambda x: uAgent2idx.get(x, uAgent2idx[unk_token])).astype(int)

def main():
    INPUT_CSV = "../../../../0-testTrainSplitter/train_split.csv"
    UNIQUE_EMB_CSV = "uAgent_embeddings_unique.csv"
    MODEL_FILE = "uAgent_embedding_model_trained.h5"
    VOCAB_FILE = "uAgent2idx.pkl"
    EMBEDDING_DIM = 64
    EPOCHS = 20
    BATCH_SIZE = 32
    df = pd.read_csv(INPUT_CSV)
    df['http_user_agent'] = (
        df['http_user_agent']
          .astype(str)
          .str.strip()
          .str.replace(r'\s+', ' ', regex=True)
          .str.lower()
    )
    unique_uAgents = df['http_user_agent'].drop_duplicates().reset_index(drop=True)
    uAgent2idx = build_vocab(unique_uAgents)
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump(uAgent2idx, f)
    ids = map_uAgents(unique_uAgents, uAgent2idx).to_numpy().reshape(-1, 1)
    labels = ids.copy()
    vocab_size = len(uAgent2idx)
    inp = Input(shape=(1,), dtype='int32')
    emb = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(inp)
    flat = Flatten()(emb)
    out = Dense(vocab_size, activation='softmax')(flat)
    clf = Model(inp, out)
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf.fit(ids, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    emb_weights = clf.get_layer(index=1).get_weights()[0]
    unique_ids = [uAgent2idx[uAgent] for uAgent in unique_uAgents]
    emb_np_unique = emb_weights[unique_ids]
    emb_cols = [f'uAgent_emb_{i}' for i in range(EMBEDDING_DIM)]
    df_embs = pd.DataFrame(emb_np_unique, columns=emb_cols)
    df_embs.to_csv(UNIQUE_EMB_CSV, index=False)
    clf.save(MODEL_FILE)
    print("✅ Unique embeddings saved to", UNIQUE_EMB_CSV)
    print("✅ Embedding model saved to", MODEL_FILE)
    print("✅ uAgent2idx mapping saved to", VOCAB_FILE)

if __name__ == "__main__":
    main()
