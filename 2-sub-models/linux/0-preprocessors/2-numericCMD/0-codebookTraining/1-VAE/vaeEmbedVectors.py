import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model

@tf.keras.utils.register_keras_serializable()
class KLDivergenceLayer(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        kl = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=1)
        self.add_loss(K.mean(kl))
        return inputs

@tf.keras.utils.register_keras_serializable(name="sampling")
def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=K.shape(mu))
    sigma = K.exp(0.5 * log_var)
    return mu + NOISE_SCALE * sigma * eps

INPUT_CSV = "../0-embedCMD/cmd_embeddings_unique.csv"
INPUT_DIM = 64
LATENT_DIM = 16
NOISE_SCALE = 0.001
EPOCHS = 40
BATCH_SIZE = 32

X = pd.read_csv(INPUT_CSV).values.astype("float32")

encoder_input = Input(shape=(INPUT_DIM,), name="encoder_input")
h_enc = Dense(32, activation="relu")(encoder_input)
mu = Dense(LATENT_DIM, name="mu")(h_enc)
log_var = Dense(LATENT_DIM, name="log_var")(h_enc)
mu, log_var = KLDivergenceLayer(name="kl")([mu, log_var])
z = tf.keras.layers.Lambda(sampling, name="z")([mu, log_var])
encoder = Model(encoder_input, [mu, log_var, z], name="encoder")

latent_input = Input(shape=(LATENT_DIM,), name="z_input")
h_dec = Dense(32, activation="relu")(latent_input)
decoder_out = Dense(INPUT_DIM, activation="linear")(h_dec)
decoder = Model(latent_input, decoder_out, name="decoder")

vae_out = decoder(z)
vae = Model(encoder_input, vae_out, name="vae")
vae.compile(optimizer="adam", loss="mse")
vae.fit(X, X, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

encoder_z = Model(encoder_input, z, name="encoder_z")
encoder_z.save("encoder_z.h5")

z_vals = encoder_z.predict(X)
df_z = pd.DataFrame(z_vals, columns=[f"z_{i}" for i in range(LATENT_DIM)])
df_z.to_csv("vae_latents_z.csv", index=False)

print("✅ Saved encoder_z.h5")
print("✅ Saved vae_latents_z.csv")