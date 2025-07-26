import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

INPUT_CSV = "../0-embedCMD/cmd_embeddings_only.csv"
INPUT_DIM = 64
LATENT_DIM = 16
ENCODER_MU_FILE = "encoder_mu.h5"
LATENTS_NOISY_CSV = "vae_latents_noisy.csv"
NOISE_SCALE = 0.001

X = pd.read_csv(INPUT_CSV).values.astype("float32")

def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=K.shape(mu))
    sigma = K.exp(0.5 * log_var)
    return mu + NOISE_SCALE * sigma * eps

class KLDivergenceLayer(Layer):
    def call(self, inputs):
        mu, log_var = inputs
        kl = -0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=1)
        self.add_loss(K.mean(kl))
        return inputs

inp = Input(shape=(INPUT_DIM,), name="encoder_input")
h = Dense(32, activation="relu")(inp)
mu = Dense(LATENT_DIM, name="mu")(h)
log_var = Dense(LATENT_DIM, name="log_var")(h)
mu, log_var = KLDivergenceLayer(name="kl_divergence")([mu, log_var])
z = Lambda(sampling, name="z")([mu, log_var])
encoder = Model(inp, [mu, log_var, z], name="encoder")

z_in = Input(shape=(LATENT_DIM,), name="z_input")
h_dec = Dense(32, activation="relu")(z_in)
out = Dense(INPUT_DIM, activation="linear")(h_dec)
decoder = Model(z_in, out, name="decoder")

vae_out = decoder(z)
vae = Model(inp, vae_out, name="vae")
vae.compile(optimizer="adam")

vae.fit(X, epochs=40, batch_size=32, verbose=1)

model_mu = Model(inp, mu, name="encoder_mu")
model_mu.save(ENCODER_MU_FILE)

encoder_z = Model(inp, z)
z_noisy = encoder_z.predict(X)
pd.DataFrame(
    z_noisy,
    columns=[f"latent_{i}" for i in range(LATENT_DIM)]
).to_csv(LATENTS_NOISY_CSV, index=False)

print("✅ Encoder μ saved to", ENCODER_MU_FILE)
print("✅ Noisy latents CSV saved to", LATENTS_NOISY_CSV)
