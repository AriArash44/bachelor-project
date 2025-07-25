from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, GRU, Dense, Dropout, GlobalAveragePooling1D

def build_bigru(seq_len, features, hidden_size, num_classes, dropout=0.2):
    inp = Input(shape=(seq_len, features))
    x = Bidirectional(GRU(hidden_size, return_sequences=True))(inp)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)
