"""
models/neural/cgan_fc.py
========================
Conditional GAN with Fully Connected layers (CGAN-FC).

Architecture (Section 2.3.2.1, Fu et al. 2019):
  Generator G(z, y) → x̃
    - Input: noise z (noise_dim,) + condition y (p*d,)
    - 3-4 FC layers with ReLU hidden, Linear output
    - Output: (q*d,) → reshaped to (q, d)

  Discriminator D(x, y) → scalar
    - Input: x (q*d,) + condition y (p*d,)
    - 3-4 FC layers with ReLU
    - Output: sigmoid probability

Loss: standard GAN minmax (Eq. 14 in paper)
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
    print("WARNING: TensorFlow not installed. Run: pip install tensorflow")


def _build_generator(noise_dim, cond_dim, out_dim, hidden_dim=256, n_layers=4):
    z_in   = keras.Input(shape=(noise_dim,), name="noise")
    cond_in = keras.Input(shape=(cond_dim,), name="condition")
    x = keras.layers.Concatenate()([z_in, cond_in])
    for _ in range(n_layers):
        x = keras.layers.Dense(hidden_dim, activation="relu")(x)
    out = keras.layers.Dense(out_dim, activation="linear")(x)
    return keras.Model([z_in, cond_in], out, name="Generator")


def _build_discriminator(sample_dim, cond_dim, hidden_dim=256, n_layers=4):
    x_in    = keras.Input(shape=(sample_dim,), name="sample")
    cond_in = keras.Input(shape=(cond_dim,),   name="condition")
    x = keras.layers.Concatenate()([x_in, cond_in])
    for _ in range(n_layers):
        x = keras.layers.Dense(hidden_dim, activation="relu")(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model([x_in, cond_in], out, name="Discriminator")


class CGANFC(BaseModel):

    def __init__(
        self,
        p: int = 10, q: int = 10, d: int = 9,
        noise_dim: int = 30,
        hidden_dim: int = 256,
        n_layers: int = 4,
        clip_value: float = 0.01,
        lr: float = 2e-4,
        batch_size: int = 64,
        epochs: int = 300,
    ):
        super().__init__("CGAN-FC", p, q, d)
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required: pip install tensorflow")
        self.noise_dim  = noise_dim
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.clip_value = clip_value
        self.lr         = lr
        self.batch_size = batch_size
        self.epochs     = epochs

        self.cond_dim   = p * d
        self.out_dim    = q * d

        self.G = _build_generator(noise_dim, self.cond_dim, self.out_dim, hidden_dim, n_layers)
        self.D = _build_discriminator(self.out_dim, self.cond_dim, hidden_dim, n_layers)

        self.g_opt = keras.optimizers.Adam(lr)
        self.d_opt = keras.optimizers.Adam(lr)
        self.bce   = keras.losses.BinaryCrossentropy()

    # ── Training step ─────────────────────────────────────────────────
    @tf.function
    def _train_step(self, real_x, cond):
        batch = tf.shape(real_x)[0]
        real_labels = tf.ones((batch, 1))
        fake_labels = tf.zeros((batch, 1))

        # ── Train Discriminator ──
        with tf.GradientTape() as tape:
            z = tf.random.normal((batch, self.noise_dim))
            fake_x = self.G([z, cond], training=False)
            # Clip weights (weight clipping for stability, from paper)
            d_real = self.D([real_x, cond], training=True)
            d_fake = self.D([fake_x, cond], training=True)
            d_loss = self.bce(real_labels, d_real) + self.bce(fake_labels, d_fake)
        grads = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grads, self.D.trainable_variables))
        # Weight clipping
        for w in self.D.trainable_variables:
            w.assign(tf.clip_by_value(w, -self.clip_value, self.clip_value))

        # ── Train Generator ──
        with tf.GradientTape() as tape:
            z = tf.random.normal((batch, self.noise_dim))
            fake_x = self.G([z, cond], training=True)
            d_fake = self.D([fake_x, cond], training=False)
            g_loss = self.bce(real_labels, d_fake)
        grads = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.G.trainable_variables))

        return d_loss, g_loss

    def fit(self, X_cond_train: np.ndarray, X_tgt_train: np.ndarray) -> "CGANFC":
        N = X_cond_train.shape[0]
        cond_flat = X_cond_train.reshape(N, -1).astype(np.float32)
        tgt_flat  = X_tgt_train.reshape(N, -1).astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((tgt_flat, cond_flat))
        dataset = dataset.shuffle(N).batch(self.batch_size)

        print(f"Training {self.name}: {self.epochs} epochs, batch={self.batch_size}")
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
        cond_flat = condition.reshape(1, -1).astype(np.float32)
        cond_rep  = np.repeat(cond_flat, n_samples, axis=0)
        z = np.random.randn(n_samples, self.noise_dim).astype(np.float32)
        fake = self.G([z, cond_rep], training=False).numpy()
        return fake.reshape(n_samples, self.q, self.d)

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.G.save(os.path.join(path, "generator"))
        self.D.save(os.path.join(path, "discriminator"))

    def load(self, path: str) -> "CGANFC":
        self.G = keras.models.load_model(os.path.join(path, "generator"))
        self.D = keras.models.load_model(os.path.join(path, "discriminator"))
        self.is_fitted = True
        return self
