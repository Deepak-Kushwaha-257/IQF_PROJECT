"""
models/neural/vae_model.py
==========================
Conditional Time VAE (based on Desai et al. 2021, extended with LSTM + conditioning).

Architecture (Section 2.3.3.1):
  Encoder:  x → [Dense+Conv+LSTM] → μ(z), log σ²(z)
  Decoder:  z + condition y → [Dense+Conv+LSTM] → x̃

Loss (ELBO, Eq. 25):
  L = -E[log p(x|z,y)] + KL(q(z|x,y) || p(z|y))

Framework: TensorFlow / Keras
Based on: github.com/abudesai/timeVAE
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from models.base_model import BaseModel

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class SamplingLayer(keras.layers.Layer):
    """Reparameterization trick: z = μ + ε·σ."""
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


class ConditionalTimeVAE(BaseModel):

    def __init__(
        self,
        p: int = 10, q: int = 10, d: int = 9,
        latent_dim: int = 30,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        kl_weight: float = 1.0,
    ):
        super().__init__("VAE", p, q, d)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required: pip install tensorflow")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr         = lr
        self.batch_size = batch_size
        self.epochs     = epochs
        self.kl_weight  = kl_weight

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.opt     = keras.optimizers.Adam(lr)

    def _build_encoder(self):
        x_in    = keras.Input(shape=(self.q, self.d), name="x_seq")
        cond_in = keras.Input(shape=(self.p, self.d), name="cond_seq")

        # Process x
        x = keras.layers.LSTM(self.hidden_dim, name="enc_lstm")(x_in)
        # Process condition and concatenate
        c = keras.layers.LSTM(self.hidden_dim // 2, name="cond_lstm")(cond_in)
        h = keras.layers.Concatenate()([x, c])
        h = keras.layers.Dense(self.hidden_dim, activation="relu")(h)

        mu      = keras.layers.Dense(self.latent_dim, name="mu")(h)
        log_var = keras.layers.Dense(self.latent_dim, name="log_var")(h)
        z       = SamplingLayer(name="z")([mu, log_var])

        return keras.Model([x_in, cond_in], [mu, log_var, z], name="Encoder")

    def _build_decoder(self):
        z_in    = keras.Input(shape=(self.latent_dim,), name="z")
        cond_in = keras.Input(shape=(self.p, self.d),   name="cond_seq")

        # Encode condition
        c = keras.layers.LSTM(self.hidden_dim // 2, name="dec_cond_lstm")(cond_in)
        h = keras.layers.Concatenate()([z_in, c])
        h = keras.layers.Dense(self.hidden_dim, activation="relu")(h)

        # Expand to sequence
        h = keras.layers.RepeatVector(self.q)(h)
        h = keras.layers.LSTM(self.hidden_dim, return_sequences=True, name="dec_lstm")(h)
        out = keras.layers.TimeDistributed(
            keras.layers.Dense(self.d, activation="linear")
        )(h)

        return keras.Model([z_in, cond_in], out, name="Decoder")

    def _vae_loss(self, x, x_recon, mu, log_var):
        # Reconstruction loss (MSE)
        recon = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_recon), axis=[1, 2]))
        # KL divergence
        kl = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        )
        return recon + self.kl_weight * kl, recon, kl

    @tf.function
    def _train_step(self, x, cond):
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder([x, cond], training=True)
            x_recon = self.decoder([z, cond], training=True)
            loss, recon, kl = self._vae_loss(x, x_recon, mu, log_var)
        grads = tape.gradient(loss, self.encoder.trainable_variables +
                              self.decoder.trainable_variables)
        self.opt.apply_gradients(
            zip(grads, self.encoder.trainable_variables +
                self.decoder.trainable_variables)
        )
        return loss, recon, kl

    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "ConditionalTimeVAE":
        N = X_cond_train.shape[0]
        dataset = (tf.data.Dataset
                   .from_tensor_slices((
                       X_tgt_train.astype(np.float32),
                       X_cond_train.astype(np.float32)
                   ))
                   .shuffle(N).batch(self.batch_size))

        print(f"Training VAE: {self.epochs} epochs")
        for epoch in range(self.epochs):
            losses = []
            for x_batch, c_batch in dataset:
                loss, recon, kl = self._train_step(x_batch, c_batch)
                losses.append(float(loss))
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} | Loss={np.mean(losses):.2f}")

        self.is_fitted = True
        return self

    def generate(self, condition: np.ndarray, n_samples: int = 251) -> np.ndarray:
        cond_rep = np.tile(condition[np.newaxis], (n_samples, 1, 1)).astype(np.float32)
        z = np.random.randn(n_samples, self.latent_dim).astype(np.float32)
        paths = self.decoder([z, cond_rep], training=False).numpy()
        return paths   # (n_samples, q, d)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.encoder.save(os.path.join(path, "encoder"))
        self.decoder.save(os.path.join(path, "decoder"))

    def load(self, path: str) -> "ConditionalTimeVAE":
        self.encoder = keras.models.load_model(
            os.path.join(path, "encoder"), custom_objects={"SamplingLayer": SamplingLayer}
        )
        self.decoder = keras.models.load_model(
            os.path.join(path, "decoder"), custom_objects={"SamplingLayer": SamplingLayer}
        )
        self.is_fitted = True
        return self
