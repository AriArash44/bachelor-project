from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Dropout

def build_simple_rnn(seq_len, features, hidden_size, num_classes, dropout=0.2):
    model = Sequential([
        Input(shape=(seq_len, features)),
        SimpleRNN(hidden_size, activation="tanh"),
        Dropout(dropout),
        Dense(num_classes, activation="softmax")
    ])
    return model
