import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Layer, Lambda, Dropout, BatchNormalization
from tensorflow.keras.models import Model

BETA = 0.5
LATENT_DIM = 16
INPUT_DIM = 64
EPOCHS = 200
BATCH_SIZE = 32

@tf.keras.utils.register_keras_serializable()
class KLDivergenceLayer(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        kl = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=1)
        self.add_loss(BETA * K.mean(kl))
        return mu, log_var

@tf.keras.utils.register_keras_serializable(name="sampling")
def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=K.shape(mu))
    sigma = K.exp(0.5 * log_var)
    return mu + 0.0001 * sigma * eps

def main():
    X = pd.read_csv("../0-embedCMD/cmd_embeddings_unique.csv").values.astype("float32")
    encoder_input = Input(shape=(INPUT_DIM,), name="encoder_input")
    x = Dropout(0.1)(encoder_input)
    x = Dense(128, activation="gelu")(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="gelu")(x)
    h_enc = BatchNormalization()(x)
    mu = Dense(LATENT_DIM, name="mu")(h_enc)
    log_var = Dense(LATENT_DIM, name="log_var")(h_enc)
    mu, log_var = KLDivergenceLayer(name="kl")([mu, log_var])
    z = Lambda(sampling, name="z")([mu, log_var])
    encoder = Model(encoder_input, [mu, log_var, z], name="encoder")
    latent_input = Input(shape=(LATENT_DIM,), name="z_input")
    x = Dense(128, activation="gelu")(latent_input)
    x = BatchNormalization()(x)
    h_dec = Dense(64, activation="gelu")(x)
    decoder_out = Dense(INPUT_DIM, activation="linear")(h_dec)
    decoder = Model(latent_input, decoder_out, name="decoder")
    vae_out = decoder(z)
    vae = Model(encoder_input, vae_out, name="vae")
    vae.compile(optimizer="adam", loss="mse")
    vae.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    mu_vals, log_var_vals, z_vals = encoder.predict(X)
    pd.DataFrame(z_vals, columns=[f"z_{i}" for i in range(LATENT_DIM)]).to_csv("vae_latents_z.csv", index=False)
    encoder_z = Model(encoder_input, z, name="encoder_z")
    encoder_z.save("encoder_z.h5")
    print("✅ Saved encoder_z.h5")
    print("✅ Saved vae_latents_z.csv")

if __name__ == "__main__":
    main()
