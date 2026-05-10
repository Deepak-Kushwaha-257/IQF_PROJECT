"""
models/neural/cgan_lstm.py
==========================
Conditional GAN with LSTM Encoder-Decoder generator (CGAN-LSTM).

Architecture (Section 2.3.2.2):
  Generator:
    - Encoder LSTM: takes condition x_{1..p} → hidden state (c1_T, a1_T)
    - Decoder LSTM: takes noise z_{1..q} + initial state from encoder → x̃_{1..q}
  Discriminator:
    - LSTM-based: processes full sequence (condition + target)

Paper uses clip_value=0.075 (tuned hyperparameter, different from CWGAN's 0.01)
Framework: TensorFlow / Keras
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


class CGANLSTM(BaseModel):

    def __init__(
        self,
        p: int = 10, q: int = 10, d: int = 9,
        noise_dim: int = 3,       # per step (paper uses 3 per step for 9-tenor)
        hidden_dim: int = 64,
        clip_value: float = 0.075,  # paper Table hyperparams
        lr: float = 2e-4,
        batch_size: int = 64,
        epochs: int = 300,
    ):
        super().__init__("CGAN-LSTM", p, q, d)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required: pip install tensorflow")

        self.noise_dim  = noise_dim
        self.hidden_dim = hidden_dim
        self.clip_value = clip_value
        self.lr         = lr
        self.batch_size = batch_size
        self.epochs     = epochs

        self.G = self._build_generator()
        self.D = self._build_discriminator()
        self.g_opt = keras.optimizers.Adam(lr)
        self.d_opt = keras.optimizers.Adam(lr)
        self.bce   = keras.losses.BinaryCrossentropy()

    def _build_generator(self):
        """Encoder LSTM → Decoder LSTM generator."""
        # Encoder
        cond_in = keras.Input(shape=(self.p, self.d), name="condition")
        enc_out, enc_h, enc_c = keras.layers.LSTM(
            self.hidden_dim, return_state=True, name="encoder_lstm"
        )(cond_in)

        # Decoder — takes noise sequence of shape (q, noise_dim)
        noise_in = keras.Input(shape=(self.q, self.noise_dim), name="noise_seq")
        dec_out = keras.layers.LSTM(
            self.hidden_dim, return_sequences=True, name="decoder_lstm"
        )(noise_in, initial_state=[enc_h, enc_c])

        # Output projection: hidden_dim → d per timestep
        output = keras.layers.TimeDistributed(
            keras.layers.Dense(self.d, activation="linear"), name="output_proj"
        )(dec_out)

        return keras.Model([cond_in, noise_in], output, name="G_LSTM")

    def _build_discriminator(self):
        """LSTM discriminator over full (p+q) sequence."""
        full_in = keras.Input(shape=(self.p + self.q, self.d), name="full_seq")
        x = keras.layers.LSTM(self.hidden_dim, name="disc_lstm")(full_in)
        x = keras.layers.Dense(64, activation="relu")(x)
        out = keras.layers.Dense(1, activation="sigmoid")(x)
        return keras.Model(full_in, out, name="D_LSTM")

    @tf.function
    def _train_step(self, real_tgt, cond_seq):
        batch = tf.shape(real_tgt)[0]
        full_real = tf.concat([cond_seq, real_tgt], axis=1)
        real_lbl  = tf.ones((batch, 1))
        fake_lbl  = tf.zeros((batch, 1))

        # Train discriminator
        with tf.GradientTape() as tape:
            noise = tf.random.normal((batch, self.q, self.noise_dim))
            fake_tgt  = self.G([cond_seq, noise], training=False)
            full_fake = tf.concat([cond_seq, fake_tgt], axis=1)
            d_real = self.D(full_real, training=True)
            d_fake = self.D(full_fake, training=True)
            d_loss = self.bce(real_lbl, d_real) + self.bce(fake_lbl, d_fake)
        grads = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grads, self.D.trainable_variables))
        for w in self.D.trainable_variables:
            w.assign(tf.clip_by_value(w, -self.clip_value, self.clip_value))

        # Train generator
        with tf.GradientTape() as tape:
            noise    = tf.random.normal((batch, self.q, self.noise_dim))
            fake_tgt = self.G([cond_seq, noise], training=True)
            full_fake = tf.concat([cond_seq, fake_tgt], axis=1)
            d_fake    = self.D(full_fake, training=False)
            g_loss    = self.bce(real_lbl, d_fake)
        grads = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.G.trainable_variables))

        return d_loss, g_loss

    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "CGANLSTM":
        N = X_cond_train.shape[0]
        cond_t = tf.constant(X_cond_train.astype(np.float32))
        tgt_t  = tf.constant(X_tgt_train.astype(np.float32))

        dataset = (tf.data.Dataset
                   .from_tensor_slices((tgt_t, cond_t))
                   .shuffle(N).batch(self.batch_size))

        print(f"Training CGAN-LSTM: {self.epochs} epochs")
        for epoch in range(self.epochs):
            d_losses, g_losses = [], []
            for real_x, cond in dataset:
                d_l, g_l = self._train_step(real_x, cond)
                d_losses.append(float(d_l))
                g_losses.append(float(g_l))
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} | "
                      f"D={np.mean(d_losses):.4f} | G={np.mean(g_losses):.4f}")

        self.is_fitted = True
        return self

    def generate(self, condition: np.ndarray, n_samples: int = 251) -> np.ndarray:
        cond_rep = np.tile(condition[np.newaxis], (n_samples, 1, 1)).astype(np.float32)
        noise    = np.random.randn(n_samples, self.q, self.noise_dim).astype(np.float32)
        fake = self.G([cond_rep, noise], training=False).numpy()
        return fake   # (n_samples, q, d)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.G.save(os.path.join(path, "generator"))
        self.D.save(os.path.join(path, "discriminator"))

    def load(self, path: str) -> "CGANLSTM":
        self.G = keras.models.load_model(os.path.join(path, "generator"))
        self.D = keras.models.load_model(os.path.join(path, "discriminator"))
        self.is_fitted = True
        return self
