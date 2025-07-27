from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    SimpleRNN,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Add
)

def build_mha_rnn(seq_len, features, hidden_size, num_classes, dropout=0.2, num_heads=2):
    inputs = Input(shape=(seq_len, features))
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=features,
        dropout=dropout
    )(inputs, inputs)
    residual = Add()([inputs, attn_output]) 
    normed = LayerNormalization()(residual)
    rnn_output = SimpleRNN(
        hidden_size,
        activation="tanh",
        return_sequences=True
    )(normed)
    pooled = GlobalAveragePooling1D()(rnn_output)
    dropped = Dropout(dropout)(pooled)
    output = Dense(num_classes, activation="softmax")(dropped)
    return Model(inputs, output)
