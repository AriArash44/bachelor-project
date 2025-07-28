import pandas as pd
import pickle
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense
from tensorflow.keras.models import Model

def build_vocab(dns_series, pad_token="<PAD>", unk_token="<UNK>"):
    unique_dnss = dns_series.dropna().tolist()
    dns2idx = {dns: idx for idx, dns in enumerate(unique_dnss, start=2)}
    dns2idx[pad_token] = 0
    dns2idx[unk_token] = 1
    return dns2idx

def map_dnss(dns_series, dns2idx, unk_token="<UNK>"):
    return dns_series.map(lambda x: dns2idx.get(x, dns2idx[unk_token])).astype(int)

def main():
    INPUT_CSV = "../../../../0-testTrainSplitter/train_split.csv"
    UNIQUE_EMB_CSV = "dns_embeddings_unique.csv"
    MODEL_FILE = "dns_embedding_model_trained.h5"
    VOCAB_FILE = "dns2idx.pkl"
    EMBEDDING_DIM = 64
    EPOCHS = 10
    BATCH_SIZE = 32
    df = pd.read_csv(INPUT_CSV)
    df['dns_query'] = (
        df['dns_query']
          .astype(str)
          .str.strip()
          .str.replace(r'\s+', ' ', regex=True)
          .str.lower()
    )
    unique_dnss = df['dns_query'].drop_duplicates().reset_index(drop=True)
    dns2idx = build_vocab(unique_dnss)
    with open(VOCAB_FILE, 'wb') as f:
        pickle.dump(dns2idx, f)
    ids = map_dnss(unique_dnss, dns2idx).to_numpy().reshape(-1, 1)
    labels = ids.copy()
    vocab_size = len(dns2idx)
    inp = Input(shape=(1,), dtype='int32')
    emb = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)(inp)
    flat = Flatten()(emb)
    out = Dense(vocab_size, activation='softmax')(flat)
    clf = Model(inp, out)
    clf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    clf.fit(ids, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    emb_weights = clf.get_layer(index=1).get_weights()[0]
    unique_ids = [dns2idx[dns] for dns in unique_dnss]
    emb_np_unique = emb_weights[unique_ids]
    emb_cols = [f'dns_emb_{i}' for i in range(EMBEDDING_DIM)]
    df_embs = pd.DataFrame(emb_np_unique, columns=emb_cols)
    df_embs.to_csv(UNIQUE_EMB_CSV, index=False)
    clf.save(MODEL_FILE)
    print("✅ Unique embeddings saved to", UNIQUE_EMB_CSV)
    print("✅ Embedding model saved to", MODEL_FILE)
    print("✅ dns2idx mapping saved to", VOCAB_FILE)

if __name__ == "__main__":
    main()
