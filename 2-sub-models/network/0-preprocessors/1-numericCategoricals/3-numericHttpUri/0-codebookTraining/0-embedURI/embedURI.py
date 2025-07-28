import pandas as pd
import pickle
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense
from tensorflow.keras.models import Model

def build_vocab(uri_series, pad_token="<PAD>", unk_token="<UNK>"):
    unique_uris = uri_series.dropna().tolist()
    uri2idx = {uri: idx for idx, uri in enumerate(unique_uris, start=2)}
    uri2idx[pad_token] = 0
    uri2idx[unk_token] = 1
    return uri2idx

def map_uris(uri_series, uri2idx, unk_token="<UNK>"):
    return uri_series.map(lambda x: uri2idx.get(x, uri2idx[unk_token])).astype(int)

def main():
    INPUT_CSV = "../../../../0-testTrainSplitter/train_split.csv"
    UNIQUE_EMB_CSV = "uri_embeddings_unique.csv"
    MODEL_FILE = "uri_embedding_model_trained.h5"
    VOCAB_FILE = "uri2idx.pkl"
    EMBEDDING_DIM = 64
    EPOCHS = 10
    BATCH_SIZE = 32
    df = pd.read_csv(INPUT_CSV)
    df['http_uri'] = (
        df['http_uri']
          .astype(str)
          .str.strip()
          .str.replace(r'\s+', ' ', regex=True)
          .str.lower()
    )
    unique_uris = df['http_uri'].drop_duplicates().reset_index(drop=True)
    uri2idx = build_vocab(unique_uris)
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump(uri2idx, f)
    ids = map_uris(unique_uris, uri2idx).to_numpy().reshape(-1, 1)
    labels = ids.copy()
    vocab_size = len(uri2idx)
    inp = Input(shape=(1,), dtype='int32')
    emb = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(inp)
    flat = Flatten()(emb)
    out = Dense(vocab_size, activation='softmax')(flat)
    clf = Model(inp, out)
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf.fit(ids, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    emb_weights = clf.get_layer(index=1).get_weights()[0]
    unique_ids = [uri2idx[uri] for uri in unique_uris]
    emb_np_unique = emb_weights[unique_ids]
    emb_cols = [f'uri_emb_{i}' for i in range(EMBEDDING_DIM)]
    df_embs = pd.DataFrame(emb_np_unique, columns=emb_cols)
    df_embs.to_csv(UNIQUE_EMB_CSV, index=False)
    clf.save(MODEL_FILE)
    print("✅ Unique embeddings saved to", UNIQUE_EMB_CSV)
    print("✅ Embedding model saved to", MODEL_FILE)
    print("✅ uri2idx mapping saved to", VOCAB_FILE)

if __name__ == "__main__":
    main()
