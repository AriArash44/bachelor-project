from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    GRU,
    Dense,
    GlobalAveragePooling1D,
    Dropout,
    MultiHeadAttention,
    Add,
    LayerNormalization
)

def build_model(
    seq_len: int,
    features: int,
    hidden_size: int,
    num_classes: int,
    dropout: float = 0.2,
    num_heads: int = 2
) -> Model:
    inp = Input(shape=(seq_len, features))
    x = Bidirectional(
        GRU(hidden_size, return_sequences=True)
    )(inp)
    att = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=hidden_size
    )(x, x)
    x = Add()([x, att])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)